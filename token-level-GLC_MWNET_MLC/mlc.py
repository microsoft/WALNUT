import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import apex
except ImportError:
    print ('No APEX installed. AMP training unavailable!')

IGNORED_INDEX = -100

def mlc_hard_ce(logit, labels, return_raw_loss=False):
    # check to see if sequence level or token level
    if logit.dim() == 2: # [bs, cls]
        loss = F.cross_entropy(logit,
                               labels,
                               ignore_index=IGNORED_INDEX,
                               reduction='none')
    elif logit.dim() == 3: # [bs, seqlen, cls]
        bs, seqlen, _ = logit.size()
        loss = F.cross_entropy(logit.view(-1, logit.size(-1)),
                               labels.view(-1),
                               ignore_index=IGNORED_INDEX,
                               reduction='none')
    if return_raw_loss:
        return loss.view(bs, seqlen)
    else:
        loss_sum = torch.sum(loss)
        loss = loss_sum / ((labels!=IGNORED_INDEX).sum())
        return loss

def mlc_soft_ce(logit, pseudo_labels, labels=None):
    # pseudo_labels is always of size [-1, cls_dim] as from LCN
    # labels is weak label only used for token task to compute label_mask
    if logit.dim() == 2: #[bs, cls]
        labels = torch.ones_like(logit[:, 0])  # Hack: a default vector of all 1s for sequence level tasks
        loss = -(pseudo_labels * F.log_softmax(logit, -1)).sum(-1) # this include IGNORED_INDEX
    elif logit.dim() == 3: #[bs, seqlen, cls]
        nclasses = logit.size(-1)
        loss = -(pseudo_labels * F.log_softmax(logit.view(-1, nclasses))).sum(-1)

        
    label_mask = (labels >= 0).view(-1)

    loss_sum = (label_mask * loss).sum()
    loss = loss_sum / (label_mask.sum())
    return loss

def _fetch_cls_emb(m):
    # m could be nn.Module or wrapped in nn.parallel.DistributedDataParallel
    # or apex.parallel.DistributedDataParallel
    if hasattr(m, 'cls_emb'): # nn.Module
        return m.cls_emb.weight
    else:
        return m.module.cls_emb.weight

#@profile
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def sync_backward(loss, opt, args, retain_graph=False): # DDP and AMP compatible backward
    if True: # no amp
        loss.backward(retain_graph=retain_graph)
    else:
        with apex.amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)

# test for gradient graph creation
def sync_autograd_debug(loss, net, opt, args, retain_graph=False, create_graph=False): # DDP and AMP compatible autograd
    if args.local_rank == -1: # single GPU
        grads = torch.autograd.grad(loss, net.parameters(), retain_graph=retain_graph, create_graph=create_graph)
    else:
        # distributed, with AMP optionally
        net.zero_grad()
        if args.amp == -1: # PyTorch DDP
            loss.backward(retain_graph=retain_graph, create_graph=create_graph)
        else: # Apex DDP
            with apex.amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph, create_graph=create_graph)

        # this assumed loss scale is 1 as when it's scaled p.grad might not be the valid grad values!
        grads = [p.grad.clone() for p in net.parameters()]

    return grads

# sync_autograd currently does not deal with gradient graph creation
def sync_autograd(loss, net, opt, args, retain_graph=False): # DDP and AMP compatible autograd
    if args.local_rank == -1: # single GPU
        grads = torch.autograd.grad(loss, net.parameters())
    else:
        # distributed, with AMP optionally
        net.zero_grad()
        if args.amp == -1: # PyTorch DDP
            loss.backward(retain_graph=retain_graph)
        else: # Apex DDP
            with apex.amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)

        # this assumed loss scale is 1 as when it's scaled p.grad might not be the valid grad values!
        grads = [p.grad.clone() for p in net.parameters()]

    return grads

#@profile
@torch.no_grad()
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    elif isinstance(opt, torch.optim.Adam):
        return update_params_adam(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported optimizer type!')


# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)

        if return_s:
            ss.append(s*eta)

    if return_s:
        return ans, ss
    else:
        return ans

# be aware that the opt state dict returns references, hence take care not to
# modify them
#@profile
def update_params_adam(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    ans = []

    if return_s:
        s = []

    beta1, beta2 = opt.defaults['betas']
    eps = opt.defaults['eps']
    wdecay = opt.defaults['weight_decay']
    amsgrad = opt.defaults['amsgrad']
    
    for i, param in enumerate(params):
        try:
            _step, _exp_avg, _exp_avg_sq = opt.state[param]['step'], opt.state[param]['exp_avg'], opt.state[param]['exp_avg_sq']
            if amsgrad:
                _max_exp_avg_sq = opt.state[param]['max_exp_avg_sq']
        except:
            _step, _exp_avg, _exp_avg_sq = 0, torch.zeros_like(param), torch.zeros_like(param)
            if amsgrad:
                _max_exp_avg_sq = torch.zeros_like(param)

        step = _step + 1
        
        dparam = grads[i] + param * wdecay
        
        exp_avg = _exp_avg * beta1 + (1-beta1) * dparam
        exp_avg_sq = torch.addcmul(_exp_avg_sq * beta2, dparam, dparam, value=1.-beta2) # change from Pytorch 1.2

        bc1 = 1. - beta1 ** step
        bc2 = 1. - beta2 ** step

        if amsgrad:
            max_exp_avg_sq = torch.max(_max_exp_avg_sq, exp_avg_sq)
            denom = max_exp_avg_sq.sqrt()/(bc2**0.5) + eps
        else:
            denom = exp_avg_sq.sqrt()/(bc2**0.5) + eps
        
        step_size = eta / bc1 
        new_param = torch.addcdiv(param, exp_avg, denom, value=-step_size) # change from Pytorch 1.2

        ans.append(new_param)

        if return_s:
            s.append(step_size * (1. - beta1) / denom.data) # .data to ensure it's not a variable # CRITICAL

    if return_s:
        return ans, s
    else:
        return ans

#@profile
def finite_difference(params, grads, eps):
    p1, p2 = [], []
    for i, param in enumerate(params):
        p1.append(param + eps * grads[i])
        p2.append(param - eps * grads[i])

    return p1, p2

def modify_parameters(net, deltas, eps):
    for param, delta in zip(net.parameters(), deltas):
        param.data.add_(eps, delta)
        #    for i, param in enumerate(net.parameters()):
        #param.data += eps * grads[i]

# ============== mlc step procedure debug with features from main model ===========
#
# METANET uses the last K-1 steps from main model and imagine one additional step ahead
# to compose a pool of actual K steps from the main model
#
# NOTE: main_net is implemented as nn.Module as usual
#       meta_net is implemented as nn.Module as usual
#       meta_net has to output PROBS, not logits!!!
#@profile
def step_hmlc_K(main_net, main_opt, 
                meta_net, meta_opt, 
                data_s, target_s, data_g, target_g,
                data_c, target_c, 
                eta, args):

    # compute gw for updating meta_net
    logit_g = main_net(data_g)
    loss_g = mlc_hard_ce(logit_g, target_g)
    gw = sync_autograd(loss_g, main_net, main_opt, args)
    
    # given current meta net, get corrected label
    logit_s, x_s_h = main_net(data_s, return_h=True)
    if logit_s.dim() == 2: # sequence level
        pseudo_target_s = meta_net(x_s_h.detach(), target_s)
        loss_s = mlc_soft_ce(logit_s, pseudo_target_s)
    else: # token level
        h_dim = x_s_h.size(-1)
        # set IGNORED_INDEX to 0, this will be counted in loss computation
        target_s__ = target_s * (target_s >= 0) 
        pseudo_target_s = meta_net(x_s_h.detach().view(-1, h_dim),
                                   target_s__.view(-1))
        loss_s = mlc_soft_ce(logit_s, pseudo_target_s, target_s)

    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = mlc_hard_ce(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2 ) / (bs1+bs2)

    # the previous backward doesn't create graph, and the following one does create graph
    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)    

    f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)
    # 2. set w as w'
    f_param = []
    for i, param in enumerate(main_net.parameters()):
        f_param.append(param.data.clone())
        param.data = f_params_new[i].data # use data only as f_params_new has graph
    
    # training loss Hessian approximation
    Hw = 1 # assume to be identity for now

    # 3. compute d_w' L_{D}(w')
    logit_g = main_net(data_g)
    loss_g  = mlc_hard_ce(logit_g, target_g)
    #gw_prime = sync_autograd(loss_g, main_net, main_opt, args)
    gw_prime = torch.autograd.grad(loss_g, main_net.parameters())

    # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
    tmp1 = [(1-Hw*dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
    gw_norm2 = (_concat(gw).norm())**2
    tmp2 = [gw[i]/gw_norm2 for i in range(len(gw))]
    gamma = torch.dot(_concat(tmp1), _concat(tmp2))

    # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
    # Refer to Eq. (29) and Eq. (32) in notes.pdf
    # NOTE: dive dparam_s by eta or not? (could affect the precision of Hv approximation)
    Lgw_prime = [ dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]     

    proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    
    sync_backward(proxy_g, meta_opt, args) # -H*L*g_w'
    # accumulate discounted iterative gradient
    for i, param in enumerate(meta_net.parameters()):
        if param.grad is not None:
            param.grad.add_(gamma * args.dw_prev[i])
            args.dw_prev[i] = param.grad.clone()

    if (args.steps+1) % (int(args.magic))==0: # T steps proceeded by main_net
        meta_opt.step()
        args.dw_prev = [0 for param in meta_net.parameters()] # 0 for previous iteration

    # modify to w, and then do actual update main_net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
        param.grad = f_param_grads[i].data
    main_opt.step()
    
    return loss_g, loss_s

def combine_loss(loss1, logit1, loss2, logit2):
    s1 = logit1.size(0)
    s2 = logit2.size(0)
    return (s1*loss1 + s2*loss2) / (s1+s2)

# loss1 has to be silver loss, 
def weighted_loss(loss1, mx1, loss2, logit2):
    assert mx1.dim() == 1
    s1 = mx1.size(0)
    s2 = logit2.size(0)
    
    loss_sum = (s1*loss1+s2*loss2)

    weight_sum = mx1.sum() + s2

    return loss_sum / weight_sum
    
# ============== l2w step procedure debug ===================
# NOTE: main_net is implemented as nn.Module as usual
def step_l2w_fast(main_net, main_opt, data_s, target_s, data_g, target_g, eta, args):
    # init eps to 0
    eps = nn.Parameter(torch.zeros_like(target_s.float()))

    # calculate current weighted loss
    logit_s = main_net(data_s)
    loss_s = (eps * F.cross_entropy(logit_s, target_s, reduction='none')).sum()
    
    # get theta grads
    # 1. update w to w'
    #param_grads = sync_autograd(loss_s, main_net, main_opt, args)
    param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)
    print (param_grads)
    
    f_param = [param.data.clone() for param in main_net.parameters()]
    #params_new = update_params(main_net.parameters(), param_grads, eta, main_opt, args)
    params_new, dparam_s = update_params(main_net.parameters(), param_grads, eta, main_opt, args, return_s=True)

    # 2. set w as w'
    for i, param in enumerate(main_net.parameters()):
        param.data = params_new[i].data # note the .data here as f_params_new might be graph from update_params

    # 3. compute d_w' L_{D}(w')
    logit_g = main_net(data_g)
    loss_g = F.cross_entropy(logit_g, target_g)

    f_param_grads_prime = torch.autograd.grad(loss_g, main_net.parameters())

    # because of dparam_s, need to scale up/down f_params_grads for proxy_g/loss_g
    # This is CRITICAL
    f_param_grads_scaled = [dparam_s[i] * param_grads[i] for i in range(len(dparam_s))]
    proxy_g = -torch.dot(_concat(f_param_grads_scaled), _concat(param_grads))
    #print ('F_grads', f_param_grads_scaled)
    #print (param_grads)
    #print (proxy_g)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]

    if args.local_rank == -1: # single GPU
        eps_grad = torch.autograd.grad(proxy_g, eps)[0]
    else:
        # distributed, with AMP optionally
        if args.amp == -1: # no amp
            proxy_g.backward()
        else: # amp
            raise NotImplementedError

        eps_grad = eps.grad.clone()

    # update eps
    w = F.relu(-eps_grad)
    
    if w.max() == 0:
        w = torch.ones_like(w)
    # normalize
    w = w / w.sum()

    logit_s = main_net(data_s)
    loss_s = (w * F.cross_entropy(logit_s, target_s, reduction='none')).sum()

    main_opt.zero_grad()
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    
    return loss_g, loss_s


# ============== mlc with l2w step procedure debug (Naive integration)===================
# NOTE: main_net is implemented as nn.Module as usual
def step_mlc_l2w_naive(main_net, main_opt, hard_loss_f,
                       meta_net, meta_opt, soft_loss_f,
                       data_s, target_s, data_g, target_g,
                       eta, args):

    # META NET AND ALPHA START
    pseudo_target_s = meta_net(data_s, target_s)

    # init eps to 0
    eps = nn.Parameter(torch.zeros_like(target_s.float()))

    # calculate current weighted loss on pseudo labels
    logit_s = main_net(data_s)
    loss_s = (eps * soft_loss_f(logit_s, pseudo_target_s, reduction='none')).sum()
    
    # get theta grads
    # 1. update w to w'
    param_grads = sync_autograd(loss_s, main_net, main_opt, args)

    params_new = update_params(main_net.parameters(), param_grads, eta, main_opt, args)

    # 2. set w as w'
    params = []
    for i, param in enumerate(main_net.parameters()):
        params.append(param.data.clone())
        param.data = params_new[i].data # use data only

    # 3. compute d_w' L_{D}(w')
    logit_g = main_net(data_g)
    loss_g = hard_loss_f(logit_g, target_g)

    params_new_grad = sync_autograd(loss_g, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = params[i]

    # change main_net parameter
    _eps = 1e-6 # 1e-3 / _concat(params_new_grad).norm # eta 1e-6 before

    # modify w to w+
    modify_parameters(main_net, params_new_grad, _eps)
    logit_s_p = main_net(data_s)
    loss_s_p = (eps * soft_loss_f(logit_s_p, pseudo_target_s, reduction='none')).sum()

    # modify w to w- (from w+)
    modify_parameters(main_net, params_new_grad, -2*_eps)
    logit_s_n = main_net(data_s)
    loss_s_n = (eps * soft_loss_f(logit_s_n, pseudo_target_s, reduction='none')).sum()

    proxy_g = -eta * (loss_s_p - loss_s_n) / (2.*_eps)

    # modify to original w
    modify_parameters(main_net, params_new_grad, _eps)

    # update meta parameters
    meta_opt.zero_grad()
    sync_backward(proxy_g, meta_opt, args) # this might not work for eps with amp
    meta_opt.step()

    eps_grad = eps.grad.clone()
    # update eps 
    w = F.relu(-eps_grad)
    
    if w.max() == 0:
        w = torch.ones_like(w)
    else:
        w = w / w.sum()
    # META NET AND ALPHA END

    # MAIN NET START
    logit_s = main_net(data_s)
    pseudo_target_s = meta_net(data_s, target_s).detach()
    loss_s = (w * soft_loss_f(logit_s, pseudo_target_s, reduction='none')).sum()

    main_opt.zero_grad()
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    
    return loss_g, loss_s


# ============== mw-net reimplementation ===================
# NOTE: main_net is implemented as nn.Module as usual
#       meta_net is VNet for instance weight 
def step_mwnet(main_net, main_opt, meta_net, meta_opt,
             data_s, target_s, data_g, target_g, 
             eta, args):

    # META NET START
    # given current meta net, get token-level instance weight
    logit_s = main_net(data_s)
    cost_s = mlc_hard_ce(logit_s, target_s, return_raw_loss=True)
    v = meta_net(cost_s.detach().unsqueeze(-1)).squeeze(-1) # [bs, seqlen, 1]

    # get weighted loss
    loss_s_sum = (v * cost_s).sum()
    loss_s = loss_s_sum / ((target_s != IGNORED_INDEX).sum())

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)
    # to test sync autograd with graph creation, doesn't work with DDP yet
    # f_param_grads = sync_autograd_debug(loss_s, main_net, main_opt, args, retain_graph=True, create_graph=True)

    # 2. pseudo update with main_opt (Computations for SGD and Adam are different!)
    f_param = [param.data.clone() for param in main_net.parameters()]
    f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)    
    for i, param in enumerate(main_net.parameters()):
        param.data = f_params_new[i].data # note the .data here as f_params_new might be graph from update_params

    # 3. compute d_w' L_{D}(w')
    logit_g = main_net(data_g)
    loss_g  = mlc_hard_ce(logit_g, target_g)

    #f_param_grads_prime = sync_autograd(loss_g, main_net, main_opt, args)
    # DEBUG: test local gradient
    f_param_grads_prime = torch.autograd.grad(loss_g, main_net.parameters())

    # because of dparam_s, need to scale up/down f_params_grads for proxy_g/loss_g
    # This is CRITICAL
    f_param_grads_scaled = [dparam_s[i] * f_param_grads[i] for i in range(len(dparam_s))]
    
    proxy_g = -torch.dot(_concat(f_param_grads_scaled), _concat(f_param_grads_prime))        

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]

    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    sync_backward(proxy_g, meta_opt, args)
    if args.grad_clip > 0:
        nn.utils.clip_grad_norm_(meta_net.parameters(), args.grad_clip)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    logit_s = main_net(data_s) # REQUIRED
    cost_s = mlc_hard_ce(logit_s, target_s, return_raw_loss=True)
    v = meta_net(cost_s.detach().unsqueeze(-1)).squeeze(-1) # [bs, seqlen, 1]

    # get weighted loss
    loss_s_sum = (v * cost_s).sum()
    loss_s = loss_s_sum / ((target_s != IGNORED_INDEX).sum())

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s, main_opt, args)
    if args.grad_clip > 0:
        nn.utils.clip_grad_norm_(main_net.parameters(), args.grad_clip)
    main_opt.step()
    # MAIN NET END

    return loss_g, loss_s


