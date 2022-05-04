import torch
import torch.nn as nn
import torch.nn.functional as F
class FullWeightModel(nn.Module):
    def __init__(self, hparams):
        super(FullWeightModel, self).__init__()
        self.hparams = hparams
        class_num = hparams.class_num
        hidden_size = hparams.hidden_size
        cls_emb_dim = hparams.cls_emb_dim
        gw_hidden_dim = hparams.gw_hidden_dim
        self.cls_emb = nn.Embedding(class_num, cls_emb_dim)
        hidden_size_input = hidden_size + cls_emb_dim

        if self.hparams.is_deeper_weight:
            self.ins_weight = nn.Sequential(
                nn.Linear(hidden_size_input, gw_hidden_dim),
                nn.Dropout(hparams.gw_dropout),
                nn.ReLU(),  # Tanh(),
                nn.Linear(gw_hidden_dim, gw_hidden_dim),
                nn.ReLU(),
                nn.Linear(gw_hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.ins_weight = nn.Sequential(
                nn.Linear(hidden_size_input, gw_hidden_dim),
                nn.Dropout(hparams.gw_dropout),
                nn.ReLU(),  # Tanh(),
                nn.Linear(gw_hidden_dim, 1),
                nn.Sigmoid()
            )


    def forward(self, x_feature, y_label, loss_s=None, y_logits=None, **kwargs):
        '''
        item_loss = 1 is just the placeholder
        '''
        x_feature = x_feature.detach()
        y_emb = self.cls_emb(y_label)
        hidden = torch.cat([y_emb, x_feature], dim=-1)
        weight = self.ins_weight(hidden)
        if loss_s is not None:
            log_loss_s = torch.mean(loss_s.detach())
            loss_s = torch.mean(weight * loss_s)
            return loss_s,  log_loss_s, weight
        return weight


class MetaNet(nn.Module):
    # def __init__(self, hx_dim, cls_dim, h_dim, num_classes, args):
    def __init__(self, hparams):
        super().__init__()

        self.args = hparams
        self.num_classes = hparams.class_num
        self.in_class = self.num_classes
        self.hdim = hparams.hidden_size
        self.cls_emb = nn.Embedding(self.in_class, hparams.cls_emb_dim)
        setattr(self.args, "tie", getattr(self.args, 'tie', False))
        self.in_dim = self.hdim + hparams.cls_emb_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.num_classes, bias=self.args.tie)
        )

        self.args.sparsemax = getattr(self.args, "sparsemax", False)
        if self.args.sparsemax:
            from sparsemax import Sparsemax
            self.sparsemax = Sparsemax(-1)

        self.init_weights()

        # if self.args.tie:
        #     print('Tying cls emb to output cls weight')
        #     self.net[-1].weight = self.cls_emb.weight

    def init_weights(self):
        nn.init.xavier_uniform_(self.cls_emb.weight)
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.xavier_normal_(self.net[4].weight)

        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()

        if self.args.tie:
            assert self.in_class == self.num_classes, 'In and out classes conflict!'
            self.net[4].bias.data.zero_()

    def get_alpha(self):
        return torch.zeros(1)

    def forward(self, hx, y):
        bs = hx.size(0)

        y_emb = self.cls_emb(y)
        hin = torch.cat([hx, y_emb], dim=-1)
        logit = self.net(hin)

        if self.args.sparsemax:
            out = self.sparsemax(logit)  # test sparsemax
        else:
            out = F.softmax(logit, -1)
        return out