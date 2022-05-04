'''
Copyright (c) Microsoft Corporation, Yichuan Li and Kai Shu.
Licensed under the MIT license.
Authors: Guoqing Zheng (zheng@microsoft.com), Yichuan Li and Kai Shu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

# no use
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

# no use
@torch.no_grad()
def update_params(params, grads, eta, opt):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt)
    elif isinstance(opt, torch.optim.Adam):
        return update_params_adam(params, grads, opt)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')

# no use
def update_params_sgd(params, grads, opt, eta):
    # supports SGD-like optimizers
    ans = []

    wdecay = opt.defaults.get('weight_decay', 0.)
    momentum = opt.defaults.get('momentum', 0.)
    # eta = opt.defaults["lr"]
    for i, param in enumerate(params):
        if grads[i] is None:
            ans.append(param)
            continue
        try:
            moment = opt.state[param]['momentum_buffer'] * momentum
        except:
            moment = torch.zeros_like(param)

        dparam = grads[i] + param * wdecay

        # eta is the learning tate
        ans.append(param - (dparam + moment) * eta)

    return ans

def update_params_adam(params, grads, opt):
    ans = []
    group = opt.param_groups[0]
    assert len(opt.param_groups) == 1
    for p, grad in zip(params, grads):
        if grad is None:
            ans.append(p)
            continue
        amsgrad = group['amsgrad']
        state = opt.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

            # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        step_size = group['lr'] / bias_correction1

        # ans.append(p.data.addcdiv(-step_size, exp_avg, denom))
        ans.append(torch.addcdiv(p, -step_size, exp_avg, denom))

    return ans


def modify_parameters(net, deltas, eps):
    for param, delta in zip([i for i in net.parameters() if i.requires_grad], deltas):
        if delta is None:
            continue
        param.data.add_(eps, delta)




# ============== l2w step procedure debug ===================
# TODO: the raw clean samples are be divided into two part: augmented data, meta-validation
# TODO: k-step look forward.
# check https://mythrex.github.io/math_behind_darts/ and https://arxiv.org/pdf/1806.09055.pdf for optimization details
def step_l2w_group_net(meta_trainer, main_net, main_opt,
                       meta_net, meta_opt,
                       clean_val_data,
                       train_weak_data, train_clean_data, **kwargs):

    # get learn rate from optimizer or scheduler
    '''
    try:
        eta_group = gw_scheduler.get_lr()
    except:
        eta_group = gw_opt.defaults.get("lr", 0)
    '''

    eta = kwargs.get("meta_lr", main_opt.defaults.get("lr", 0))

    # calculate current weighted loss
    # ATTENTION: loss_s shape: [bs * K, 1]
    y_weak = train_weak_data['y']
    train_weak_data['is_reduction'] = False
    outputs_s = main_net(**train_weak_data)
    # hidden, logit, loss
    s_feature = outputs_s[0]
    loss_s = outputs_s[-1]
    logits_s = outputs_s[1]
    train_weak_data['y_label'] = y_weak
    loss_s, _ = meta_net(x_feature=s_feature, loss_s=loss_s,
                         y_logits=logits_s, **train_weak_data)
    loss_train = main_net(**train_clean_data)[-1]
    loss_s = (loss_s + loss_train) / 2


    # get theta grads
    # 1. update w to w'
    param_grads = torch.autograd.grad(loss_s, [i for i in main_net.parameters() if i.requires_grad], allow_unused=True)

    # 2. set w as w'
    main_params_raw = [param.data.clone() for param in [i for i in main_net.parameters() if i.requires_grad]]
    for i, param in enumerate([i for i in main_net.parameters() if i.requires_grad]):
        if param_grads[i] is not None:
            param.data.sub_(eta * param_grads[i])

    # 3. compute d_w' L_{D}(w')
    loss_g = main_net(**clean_val_data)[-1]

    params_new_grad = torch.autograd.grad(loss_g, [i for i in main_net.parameters() if i.requires_grad], allow_unused=True)

    # 4. revert from w' to w for main net
    for i, param in enumerate([i for i in main_net.parameters() if i.requires_grad]):
        param.data = main_params_raw[i]

    # change main_net parameter
    _eps = 1e-6  # 1e-3 / _concat(params_new_grad).norm # eta 1e-6 before

    # modify w to w+
    modify_parameters(main_net, params_new_grad, _eps)
    outputs_s_p = main_net(**train_weak_data)
    loss_s_p = outputs_s_p[-1]
    logits_s_p = outputs_s_p[1]
    s_p_feature = outputs_s_p[0]

    loss_s_p, _ = meta_net(x_feature=s_p_feature, loss_s=loss_s_p,
                           y_logits=logits_s_p, **train_weak_data)
    loss_train = main_net(**train_clean_data)[-1]
    loss_s_p = (loss_s_p + loss_train) / 2
    grads_s_p = torch.autograd.grad(loss_s_p, meta_net.parameters())

    # modify w to w- (from w+)
    modify_parameters(main_net, params_new_grad, -2 * _eps)
    outputs_s_n = main_net(**train_weak_data)
    loss_s_n = outputs_s_n[-1]
    logits_s_n = outputs_s_n[1]
    s_n_feature = outputs_s_n[0]
    loss_s_n, _ = meta_net(x_feature=s_n_feature, loss_s=loss_s_n,
                           y_logits=logits_s_n, **train_weak_data)
    loss_train = main_net(**train_clean_data)[-1]
    loss_s_n = (loss_s_n + loss_train) / 2
    grads_s_n = torch.autograd.grad(loss_s_n, meta_net.parameters())
    # proxy_g = -eta * (loss_s_p - loss_s_n) / (2. * _eps)
    grad_g = [(i-j).div_(2*_eps) for i, j in zip(grads_s_p, grads_s_n)]


    # update gw
    meta_opt.zero_grad()
    # set the gradient
    # TODO: gradient clip
    for (name, param_gw), grad in zip(meta_net.named_parameters(), grad_g):
        param_gw.grad = grad
    torch.nn.utils.clip_grad_norm_(meta_net.parameters(), 5)
    meta_opt.step()
    if meta_trainer.trainer.global_step % 25 == 0:  # don't make the tf file huge
        for name, p in meta_net.named_parameters():
            grad = p.grad
            meta_trainer.logger.experiment.add_histogram(
            tag=name, values=grad, global_step=meta_trainer.trainer.global_step)

    # call scheduler for gw if applicable here
    # modify to original w
    modify_parameters(main_net, params_new_grad, _eps)
    outputs_s = main_net(**train_weak_data)
    loss_s = outputs_s[-1]
    logits_s = outputs_s[1]
    s_feature = outputs_s[0]
    loss_s, instance_weight = meta_net(x_feature=s_feature, loss_s=loss_s,
                                       y_logits=logits_s, **train_weak_data)

    # write the group weight and instance weight

    # mean reduction
    loss_train = main_net(**train_clean_data)[-1]
    loss_final = (loss_s + loss_train) / 2

    main_opt.zero_grad()
    meta_trainer.manual_backward(loss_final)

    # loss_s.backward()
    main_opt.step()

    return loss_g, loss_s, loss_train, loss_final, instance_weight


def step_l2w_group_net_previous(meta_trainer, main_net, main_opt,
                                meta_net, meta_opt,
                                val_clean_data,
                                train_weak_data, train_clean_data):

    # get learn rate from optimizer or scheduler
    '''
    try:
        eta_group = gw_scheduler.get_lr()
    except:
        eta_group = gw_opt.defaults.get("lr", 0)
    '''

    eta = main_opt.defaults.get("lr", 0)

    # calculate current weighted loss
    # ATTENTION: loss_s shape: [bs * K, 1]
    y_weak = train_weak_data['y']
    train_weak_data['is_reduction'] = False
    outputs_s = main_net(**train_weak_data)
    # hidden, logit, loss
    s_feature = outputs_s[0]
    loss_s = outputs_s[-1]
    loss_s, _ = meta_net(s_feature, y_weak, loss_s)
    loss_train = main_net(**train_clean_data)[-1]
    loss_s = (loss_s + loss_train) / 2


    # get theta grads
    # 1. update w to w'
    param_grads = torch.autograd.grad(loss_s, [i for i in main_net.parameters() if i.requires_grad], allow_unused=True)

    # 2. set w as w'
    params = [param.data.clone() for param in [i for i in main_net.parameters() if i.requires_grad]]
    for i, param in enumerate([i for i in main_net.parameters() if i.requires_grad]):
        if param_grads[i] is not None:
            param.data.sub_(eta * param_grads[i])

    # 3. compute d_w' L_{D}(w')
    loss_g = main_net(**val_clean_data)[-1]

    params_new_grad = torch.autograd.grad(loss_g, [i for i in main_net.parameters() if i.requires_grad], allow_unused=True)

    # 4. revert from w' to w for main net
    for i, param in enumerate([i for i in main_net.parameters() if i.requires_grad]):
        param.data = params[i]

    # change main_net parameter
    _eps = 1e-6  # 1e-3 / _concat(params_new_grad).norm # eta 1e-6 before

    # modify w to w+
    modify_parameters(main_net, params_new_grad, _eps)
    outputs_s_p = main_net(**train_weak_data)
    loss_s_p = outputs_s_p[-1]
    s_p_feature = outputs_s_p[0]
    loss_s_p, _ = meta_net(s_p_feature, y_weak, loss_s_p)
    loss_train = main_net(**train_clean_data)[-1]
    loss_s_p = (loss_s_p + loss_train) / 2

    # modify w to w- (from w+)
    modify_parameters(main_net, params_new_grad, -2 * _eps)
    outputs_s_n = main_net(**train_weak_data)
    loss_s_n = outputs_s_n[-1]
    s_n_feature = outputs_s_n[0]
    loss_s_n, _ = meta_net(s_n_feature, y_weak, loss_s_n)

    loss_train = main_net(**train_clean_data)[-1]
    loss_s_n = (loss_s_n + loss_train) / 2

    proxy_g = -eta * (loss_s_p - loss_s_n) / (2. * _eps)

    # modify to original w
    modify_parameters(main_net, params_new_grad, _eps)


    # update gw
    # gw_opt.zero_grad()
    meta_trainer.manual_backward(proxy_g)
    # proxy_g.backward()
    if meta_trainer.trainer.global_step % 25 == 0:  # don't make the tf file huge
        for name, p in meta_net.named_parameters():
            grad = p.grad
            meta_trainer.logger.experiment.add_histogram(
            tag=name, values=grad, global_step=meta_trainer.trainer.global_step)
    meta_opt.step()

    # call scheduler for gw if applicable here
    outputs_s = main_net(**train_weak_data)
    loss_s = outputs_s[-1]
    s_feature = outputs_s[0]
    loss_s, instance_weight = meta_net(s_feature, y_weak, loss_s)

    # write the group weight and instance weight

    # mean reduction
    loss_train = main_net(**train_clean_data)[-1]
    loss_final = (loss_s + loss_train) / 2

    meta_trainer.manual_backward(loss_final)
    # main_opt.zero_grad()
    # loss_s.backward()
    main_opt.step()

    return loss_g, loss_s, loss_train, loss_final, instance_weight


def step_l2r(meta_trainer, main_net, main_opt, val_clean_data, train_weak_data, train_clean_data=None, **kwargs):

    main_scheduler = kwargs['main_scheduler']
    train_weak_data['is_reduction'] = False

    eta = kwargs['meta_opt'].defaults.get("lr", 0)
    # eta = main_opt.defaults.get("lr", 0)

    eps = nn.Parameter(torch.zeros_like(train_weak_data['y'].float()))
    eps = eps.view(-1)

    # calculate current weighted loss
    main_net.train()

    loss_s = main_net(**train_weak_data)[-1]
    # {reduction: "none"} in s_inputs

    loss_s = (eps * loss_s).sum()
    if train_clean_data is not None:
        loss_train = main_net(**train_clean_data)[-1]
        loss_s = (loss_train + loss_s) / 2

    # get theta grads
    # 1. update w to w'
    param_grads = torch.autograd.grad(loss_s, [p for p in main_net.parameters() if p.requires_grad], allow_unused=True)

    params_new = update_params_sgd([p for p in main_net.parameters() if p.requires_grad], param_grads, main_opt, eta)

    # 2. set w as w'
    params = []
    for i, param in enumerate([p for p in main_net.parameters() if p.requires_grad]):
        params.append(param.data.clone())
        param.data = params_new[i].data  # use data only
    # 3. compute d_w' L_{D}(w')

    loss_g = main_net(**val_clean_data)[-1]

    params_new_grad = torch.autograd.grad(loss_g, [p for p in main_net.parameters() if p.requires_grad], allow_unused=True)

    # 4. revert from w' to w for main net
    for i, param in enumerate([p for p in main_net.parameters() if p.requires_grad]):
        param.data = params[i]

    # change main_net parameter
    _eps = 1e-6  # 1e-3 / _concat(params_new_grad).norm # eta 1e-6 before

    # modify w to w+
    modify_parameters(main_net, params_new_grad, _eps)

    loss_s_p = main_net(**train_weak_data)[-1]
    loss_s_p = (eps * loss_s_p).sum()
    if train_clean_data is not None:
        loss_train_p = main_net(**train_clean_data)[-1]
        loss_s_p = (loss_s_p + loss_train_p) / 2

    # modify w to w- (from w+)
    modify_parameters(main_net, params_new_grad, -2 * _eps)
    loss_s_n = main_net(**train_weak_data)[-1]
    loss_s_n = (eps * loss_s_n).sum()
    if train_clean_data is not None:
        loss_train_n = main_net(**train_clean_data)[-1]
        loss_s_n = (loss_train_n + loss_s_n)

    proxy_g = -eta * (loss_s_p - loss_s_n) / (2. * _eps)

    # modify to original w
    modify_parameters(main_net, params_new_grad, _eps)
    eps_grad = torch.autograd.grad(proxy_g, eps, allow_unused=True)[0]

    # update eps
    w = F.relu(-eps_grad)

    if w.max() == 0:
        w = torch.ones_like(w)
    else:
        w = w / w.sum()

    loss_s = main_net(**train_weak_data)[-1]
    loss_s = (w * loss_s).sum()
    if train_clean_data is not None:
        loss_train = main_net(**train_clean_data)[-1]
        loss_s = (loss_s + loss_train) / 2

    if train_clean_data is not None:
        loss_s += main_net(**train_clean_data)[-1]

    # main_opt.zero_grad()
    main_net.zero_grad()

    meta_trainer.manual_backward(loss_s)
    main_opt.step()

    # loss_val, loss_s, log_loss_s, loss_train_clean, loss_final, instance_weight
    # -1 is the placeholder
    return loss_g, loss_s, loss_g, loss_s, loss_g + loss_s, None