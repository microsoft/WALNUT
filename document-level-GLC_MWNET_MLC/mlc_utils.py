import torch
import torch.nn.functional as F

def save_checkpoint(state, filename):
    torch.save(state, filename)

def pseudo_eval(model): # this turns dropout off, but leavs batchnorm on -> used by meta models
    assert isinstance(model, torch.nn.Module), 'Model has to be of type nn.Module!'
    
    for m in model.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd): # turns off all dropout
            m.eval()

class DummyScheduler(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
                       
        return lrs

    def step(self, epoch=None):
        pass

def tocuda(data, device=torch.device("cuda:0")):
    if type(data) is list:
        if len(data) == 1:
            if torch.cuda.is_available():
                return data[0].to(device)
            else:
                return data[0]
        else:
            if torch.cuda.is_available():
                return [x.to(device) for x in data]
            else:
                return [x for x in data]

    else:
        if torch.cuda.is_available():
            return data.to(device)
        else:
            return data

def net_grad_norm_max(model, p):
    grad_norms = [x.grad.data.norm(p).item() for x in model.parameters()]
    return max(grad_norms)

def clone_parameters(model):
    assert isinstance(model, torch.nn.Module), 'Wrong model type'

    params = model.named_parameters()

    f_params_dict = {}
    f_params = []
    for idx, (name, param) in enumerate(params):
        new_param = torch.nn.Parameter(param.data.clone())
        f_params_dict[name] = idx
        f_params.append(new_param)

    return f_params, f_params_dict

# target differentiable version of F.cross_entropy        
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

# masked cross entropy for sequence tagging tasks
def masked_cross_entropy(logit, labels, weights=None):
    loss = F.cross_entropy(logit.view(-1, logit.size(-1)),
                           labels.reshape(-1),
                           ignore_index=IGNORED_INDEX,
                           reduction='none')
        
    loss_sum = torch.sum(loss)
    loss = loss_sum / (labels!=IGNORED_INDEX).sum()
    return loss

# test code for soft_cross_entropy
if __name__ == '__main__':
    K = 100
    for _ in range(10000):
        y = torch.randint(K, (100,))
        y_onehot = F.one_hot(y, K).float()
        logits = torch.randn(100, K)

        l1 = F.cross_entropy(logits, y)
        l2 = soft_cross_entropy(logits, y_onehot)

        print (l1.item(), l2.item(), '%.5f' %(l1-l2).abs().item())
        
