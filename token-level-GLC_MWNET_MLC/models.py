import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ThreeLayerNet(nn.Module):
    def __init__(self, in_dim, h_dim, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_classes)
            )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.main[0].weight)
        self.main[0].bias.data.zero_()
        nn.init.xavier_normal_(self.main[2].weight)
        self.main[2].bias.data.zero_()
        nn.init.xavier_normal_(self.main[4].weight)        
        self.main[4].bias.data.zero_()

    def forward(self, x, return_h=False):
        if return_h:
            x_h = self.main[:-1](x)
            out = self.main[-1:](x_h)
            return x_h, out
        else:
            return self.main(x)

# //////////////////////// defining graph ////////////////////////
# implemented with nn.Module, for pre-train use
# ///////////////////////////////////////////////////////////////
class WordAveragingLinear(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.out = nn.Linear(emb_dim, num_classes)
        
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_normal_(self.out.weight)
        self.out.bias.data.zero_()

    def forward(self, x, return_h=False):
        if return_h:
            x_h = self.embedding(x).mean(1)
            out = self.out(x_h)
            return x_h, out
        else:
            return self.out(self.embedding(x).mean(1))


    
# ///////////////////////////////////////////////////////////////
class LSTMNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, h_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, h_dim)
        self.out = nn.Linear(h_dim, num_classes)
        self.drop = nn.Dropout(0.2)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_normal_(self.out.weight)
        self.out.bias.data.zero_()

    def forward(self, x, return_h=False):
        emb = self.embedding(x).permute(1,0,2)
        
        # use final output to encode x
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(emb)
        #return self.out(self.drop(h.squeeze(0)))
        out = self.out(rnn_out[-1])

        if return_h:
            return rnn_out, self.drop(out)
        else:
            return self.drop(out)


class HFSC(nn.Module): 
    def __init__(self, model_name, num_classes):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.model.classifier.weight)
        #self.model.classifier.weight.data.normal_(mean=0, std=0.02)
        self.model.classifier.bias.data.zero_()

    def forward(self, x, return_h=False):
        # x is actually a tuple of (data, mask)
        data, mask = x
        outputs = self.model(data, attention_mask=mask, output_hidden_states=return_h)
        logit = outputs.logits
        if return_h:
            return logit, outputs.hidden_states[-1][:, 0, :] # only for [CLS]
        else:
            return logit

class HFTC(nn.Module): 
    def __init__(self, model_name, num_classes):
        super().__init__()
        from transformers import AutoModelForTokenClassification
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.model.classifier.weight)
        #self.model.classifier.weight.data.normal_(mean=0, std=0.02)
        self.model.classifier.bias.data.zero_()

    def forward(self, x, return_h=False):
        # x is actually a tuple of (data, mask)
        data, mask = x
        outputs = self.model(data, attention_mask=mask, output_hidden_states=return_h)
        logit = outputs.logits
        
        if return_h:
            return logit, outputs.hidden_states[-1] # for all tokens, [bs, seqlen, dim]
        else:
            return logit
        

class LSTM_text(nn.Module):
    def __init__(self, num_classes, h_dim, embed_weight: np.ndarray, is_ner):
        super(LSTM_text, self).__init__()
        C = num_classes
        # one for padded word
        V = embed_weight.shape[0]
        D = embed_weight.shape[1]

        self.embed = nn.Embedding(V, D, padding_idx=0).from_pretrained(torch.FloatTensor(embed_weight), freeze=False)

        self.lstm = nn.LSTM(D, hidden_size=h_dim//2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.head_count = 1
        # self.final_prediction = nn.Linear(self.args.hidden_size, C*self.head_count)
        # self.fc2 = nn.Linear(len(Ks) * Co, self.args.hidden_size)

        self.final_prediction = nn.Linear(h_dim, C)

        self.class_num = num_classes

        self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.loss_reduction = nn.CrossEntropyLoss(ignore_index=-1)

        self.pad_idx = 0
        self.is_ner = is_ner
        self.pooling_fn = "mean"


    def forward(self, x, return_h=False):
        if type(x) is list or type(x) is tuple:
            x, attention_mask = x
            x_length = torch.sum(attention_mask, dim=1)
        else:
            x_length = torch.sum(x != self.pad_idx, dim=1)
        embed_x = self.embed(x)
        sorted_seq_len, permIdx = x_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_embed_x = embed_x[permIdx]
        packed_words = pack_padded_sequence(sorted_embed_x, sorted_seq_len.cpu(), True)
        self.lstm.flatten_parameters()
        lstm_out, (hidden, cell) = self.lstm(packed_words, None)
        if self.is_ner:
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)
            feature_out = self.dropout(lstm_out)
        else:
            hidden = hidden.transpose(1, 0)
            # take the last layer hidden state.
            hidden = hidden[:, -2:].reshape(hidden.shape[0], -1)
            feature_out = self.dropout(hidden)
        # reorder the elements
        feature_out = feature_out[recover_idx]
        logit = self.final_prediction(feature_out)
        if return_h:
            return logit, feature_out
        else:
            return logit
        # logit = logit.reshape(-1, self.class_num)
        # y = y.reshape(-1, )
        # output = (feature_out.detach(), logit,)
        # if y is not None:
        #     if is_reduction:
        #         loss = self.loss_reduction(logit, y)
        #     else:
        #         loss = self.loss_no_reduction(logit, y)
        #     output += (loss, )
        # return output


class LSTM_text_TC(nn.Module):
    def __init__(self, num_classes, h_dim, embed_weight: np.ndarray, is_ner):
        super(LSTM_text_TC, self).__init__()
        C = num_classes
        self.num_labels = num_classes
        # one for padded word
        V = embed_weight.shape[0]
        D = embed_weight.shape[1]

        self.embed = nn.Embedding(V, D, padding_idx=0).from_pretrained(torch.FloatTensor(embed_weight), freeze=False)

        self.lstm = nn.LSTM(D, hidden_size=h_dim//2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.head_count = 1
        # self.final_prediction = nn.Linear(self.args.hidden_size, C*self.head_count)
        # self.fc2 = nn.Linear(len(Ks) * Co, self.args.hidden_size)

        self.final_prediction = nn.Linear(h_dim, C)

        self.class_num = num_classes

        # self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        def myCrossEntropyLoss(outputs, labels):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            """
            import pdb; pdb.set_trace()
            labels = labels.view(-1)
            mask = (labels >= 0).float()
            labels[labels < 0] = 0

            outputs = outputs.view(-1, outputs.shape[-1])
            outputs = outputs[range(outputs.shape[0]), labels] * mask
            # outputs = outputs * mask

            # pick the values corresponding to labels and multiply by mask
            #
            """
            # loss = loss_fct(outputs, labels)
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            return loss
        self.loss_reduction = myCrossEntropyLoss
        #self.loss_reduction = nn.CrossEntropyLoss(ignore_index=-1)
        #self.loss_reduction = self.loss_fn

        self.pad_idx = 0
        self.is_ner = is_ner
        self.pooling_fn = "mean"
        self.pad_len = 200


    def forward(self, input_ids, attention_mask=None, labels=None, return_h=False):
        x=input_ids
        #if type(x) is list or type(x) is tuple: # giannis: also consider type=tuple
        if type(x) is list or type(x) is tuple:
            x, attention_mask = x
            x_length = torch.sum(attention_mask, dim=1)
        else:
            #import pdb; pdb.set_trace()
            x_length = torch.sum(torch.tensor(x) != self.pad_idx, dim=1)
        #import pdb; pdb.set_trace()
        device=torch.device('cuda:0') # giannis FIXME: provide the right device
        x = torch.tensor(x).to(device)
        embed_x = self.embed(x)
        sorted_seq_len, permIdx = x_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_embed_x = embed_x[permIdx]

        sorted_seq_len[sorted_seq_len==0] = 1 # FIXME: fix issue of empty sequences (len==0)
        packed_words = pack_padded_sequence(sorted_embed_x, sorted_seq_len.cpu(), True)

        self.lstm.flatten_parameters()
        lstm_out, (hidden, cell) = self.lstm(packed_words, None)

        if self.is_ner:
            #lstm_out = pad_packed_sequence(lstm_out, batch_first=True)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.pad_len) # giannis: fixed bug
            feature_out = self.dropout(lstm_out)
        else:
            hidden = hidden.transpose(1, 0)
            # take the last layer hidden state.
            hidden = hidden[:, -2:].reshape(hidden.shape[0], -1)
            feature_out = self.dropout(hidden)
        # reorder the elements
        feature_out = feature_out[recover_idx]
        logit = self.final_prediction(feature_out)

        '''
        # will also add loss: https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForTokenClassification
        # import pdb; pdb.set_trace()
        if labels is not None:
            # loss = self.loss_fn(logit, labels)
            loss = self.loss_reduction(logit, labels)
        #output = (feature_out.detach(), logit,)
        #output = (logit)
        #output += (loss,)
        #output = (logit, loss)
        output = (loss, logit)
        return output
        '''

        if return_h:
            return logit, feature_out
        else:
            return logit
        # logit = logit.reshape(-1, self.class_num)
        # y = y.reshape(-1, )
        # output = (feature_out.detach(), logit,)
        # if y is not None:
        #     if is_reduction:
        #         loss = self.loss_reduction(logit, y)
        #     else:
        #         loss = self.loss_no_reduction(logit, y)
        #     output += (loss, )
        # return output

    def loss_fn(self, outputs, labels):

        #reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.view(-1)

        #mask out 'PAD' tokens
        mask = (labels >= 0).float()

        #the number of tokens is the sum of elements in mask
        #num_tokens = int(torch.sum(mask).data[0])
        num_tokens = int(torch.sum(mask).data)

        outputs = outputs.view(-1, outputs.shape[-1])
        labels[labels<0] = 0
        #pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels]*mask

        #cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs)/num_tokens
        
