# https://github.com/microsoft/MLC
import torch
import torch.nn.functional as F

def soft_cross_entropy(logit, pseudo_target, reduction='mean'):
    loss = -(pseudo_target * F.log_softmax(logit, -1)).sum(-1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError('Invalid reduction: %s' % reduction)


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


@torch.no_grad()
def update_params(params, grads, eta, opt, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')


# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay  # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. - dampening)  # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment  # s= 1+momentum*(1.-dampening)
                s = 1 + momentum * (1. - dampening)
            else:
                dparam = moment  # s=1.-dampening
                s = 1. - dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam * eta)

        if return_s:
            ss.append(s * eta)

    if return_s:
        return ans, ss
    else:
        return ans

# ============== mlc step procedure debug with features (gradient-stopped) from main model ===========
#
# METANET uses the last K-1 steps from main model and imagine one additional step ahead
# to compose a pool of actual K steps from the main model
#
#
def step_hmlc_K(meta_trainer, main_net, main_opt,
                meta_net, meta_opt,
                val_clean_data, train_weak_data, train_clean_data=None, **kwargs):
    eta = kwargs['main_scheduler'].get_lr()[0]
    # compute gw for updating meta_net
    loss_g = main_net(**val_clean_data)[-1]
    gw = torch.autograd.grad(loss_g, [p for p in main_net.parameters() if p.requires_grad], allow_unused=True)
    args = meta_trainer.hparams

    # given current meta net, get corrected label
    train_weak_data['is_reduction'] = False
    outputs_s = main_net(**train_weak_data)
    # hidden, logit, loss
    s_feature = outputs_s[0]
    s_logits = outputs_s[1]
    y_weak = train_weak_data['y']
    train_weak_data['y_label'] = y_weak
    pseudo_target_s = meta_net(s_feature.detach(), y_weak)
    loss_s = soft_cross_entropy(s_logits, pseudo_target_s)
    # loss_s_aug, log_loss_s, instance_weight = meta_net(x_feature=s_feature,
    #                                                    loss_s=loss_s_list, y_logits=s_logits, **train_weak_data)

    if train_clean_data is not None:
        # the clean training dataset
        # TODO: balance the loss of clean and train
        # bs1 = target_s.size(0)
        # bs2 = target_c.size(0)
        loss_train_clean = main_net(**train_clean_data)[-1]
        loss_final = (loss_s + loss_train_clean) / 2
    else:
        # for the output length padding
        loss_train_clean = loss_s
        loss_final = loss_s

    f_param_grads = torch.autograd.grad(loss_final, [p for p in main_net.parameters() if p.requires_grad], create_graph=True,
                                        allow_unused=True)

    f_params_new, dparam_s = update_params([p for p in main_net.parameters() if p.requires_grad], f_param_grads, eta, main_opt, return_s=True)
    # 2. set w as w'
    f_param = []
    for i, param in enumerate(filter(lambda p: p.requires_grad, main_net.parameters())):
        f_param.append(param.data.clone())
        param.data = f_params_new[i].data  # use data only as f_params_new has graph

    # training loss Hessian approximation
    Hw = 1  # assume to be identity

    # 3. compute d_w' L_{D}(w')
    loss_g = main_net(**val_clean_data)[-1]
    gw_prime = torch.autograd.grad(loss_g, [p for p in main_net.parameters() if p.requires_grad], allow_unused=True)

    # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
    tmp1 = [(1 - Hw * dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
    gw_norm2 = (_concat(gw).norm()) ** 2
    tmp2 = [gw[i] / gw_norm2 for i in range(len(gw))]
    gamma = torch.dot(_concat(tmp1), _concat(tmp2))

    # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
    Lgw_prime = [dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]

    proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

    # back prop on alphas
    meta_opt.zero_grad()
    meta_trainer.manual_backward(proxy_g)

    # accumulate discounted iterative gradient
    for i, param in enumerate(filter(lambda p: p.requires_grad, meta_net.parameters())):
        if param.grad is not None:
            param.grad.add_(gamma * meta_trainer.dw_prev[i])
            meta_trainer.dw_prev[i] = param.grad.clone()

    if (meta_trainer.global_step + 1) % (args.gradient_steps) == 0:  # T steps proceeded by main_net
        meta_opt.step()
        # log the meta grad
        if meta_trainer.trainer.global_step % 100 == 0:  # don't make the tf file huge
            for name, p in meta_net.named_parameters():
                if p.requires_grad is False:
                    continue
                grad = p.grad
                if grad is not None:
                    try:
                        meta_trainer.logger.experiment.add_histogram(
                            tag="meta/"+name, values=grad, global_step=meta_trainer.trainer.global_step)
                        meta_trainer.logger.experiment.add_scalar(
                            "GradientNormMeta/" + name, grad.norm(2), meta_trainer.trainer.global_step)
                    except:
                        continue
            meta_trainer.dw_prev = [0 for p in meta_net.parameters() if p.requires_grad]  # 0 to reset

    # modify to w, and then do actual update main_net
    for i,  param in enumerate(filter(lambda p: p.requires_grad, main_net.parameters())):
        param.data = f_param[i]
        param.grad = f_param_grads[i].data
    # log the model gradient
    if meta_trainer.global_step % 100 == 0:
        for n, p in main_net.named_parameters():
            if p.requires_grad is False:
                continue
            grad = p.grad
            if grad is not None:
                try:
                    meta_trainer.logger.experiment.add_histogram(
                        tag="Main/" + n, values=grad, global_step=meta_trainer.trainer.global_step)
                    meta_trainer.logger.experiment.add_scalar(
                        "GradientNormMain/"+n, grad.norm(2), meta_trainer.trainer.global_step)
                except:
                    continue

    main_opt.step()


    # these return elements only for log.
    return loss_g, loss_s, loss_s, loss_train_clean, loss_final, None