import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MetaNet(nn.Module):
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes, args):
        super().__init__()

        self.args = args

        self.num_classes = num_classes        
        self.in_class = self.num_classes 
        self.hdim = h_dim
        self.cls_emb = nn.Embedding(self.in_class, cls_dim)

        in_dim = hx_dim + cls_dim
        out_dim = num_classes + args.skip
        
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, out_dim, bias=(not self.args.tie)) 
        )

        if self.args.sparsemax:
            from sparsemax import Sparsemax
            self.sparsemax = Sparsemax(-1)

        self.init_weights()

        if self.args.tie:
            print ('Tying cls emb to output cls weight')
            assert cls_dim == self.hdim
            self.net[-1].weight = self.cls_emb.weight
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.cls_emb.weight)
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.xavier_normal_(self.net[4].weight)

        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()

        if not self.args.tie:
            assert self.in_class == self.num_classes, 'In and out classes conflict!'
            self.net[4].bias.data.zero_()

    def get_alpha(self):
        return self.alpha if self.args.skip else torch.zeros(1)

    def forward(self, hx, y):
        bs = hx.size(0)
        y_emb = self.cls_emb(y)
        hin = torch.cat([hx, y_emb], dim=-1)
        logit = self.net(hin)
        if self.args.skip:
            alpha = torch.sigmoid(logit[:, self.num_classes:])
            self.alpha = alpha.mean()
            logit = logit[:, :self.num_classes]

        if self.args.sparsemax:
            out = self.sparsemax(logit) # test sparsemax
        else:
            out = F.softmax(logit, -1)

        if self.args.skip:
            out = alpha * F.one_hot(y, self.num_classes).type_as(logit) + (1.-alpha) * out

        return out

# Meta-WN (As in original paper)
class VNet_(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(VNet_, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

# Meta-WN which has similar arch to MLC meta net    
class VNet(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(VNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))
