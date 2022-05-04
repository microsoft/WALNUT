import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig, DistilBertConfig, DistilBertModel, AutoModel, AutoConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class CNN_Text(nn.Module):
    def __init__(self, args, use_roberta_wordembed=True):
        super(CNN_Text, self).__init__()
        self.args = args
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = list(map(int, args.kernel_sizes.split(",")))

        disbert = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        V = disbert.vocab_size
        D = disbert.hidden_size

        dis_bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=disbert)
        embedding_weight = dis_bert_model.get_input_embeddings().weight
        # ATTENTION: the word embedding is freezed to consistent with previous work
        self.embed = nn.Embedding(V, D).from_pretrained(embedding_weight, freeze=getattr(self.args, "is_freeze", True))
        del dis_bert_model
        del embedding_weight

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.cnn_drop_out)

        self.head_count = getattr(self.args, "head_count", 1)

        self.final_prediction = nn.Linear(len(Ks) * Co, C * self.head_count)

        self.class_num = args.class_num

        self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.loss_reduction = nn.CrossEntropyLoss(ignore_index=-1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, y=None, is_reduction=True, **kwargs):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        hidden = self.dropout(x)  # (N, len(Ks)*Co)
        # hidden = self.fc2(x)
        logit = self.final_prediction(hidden)  # (N, C * num_head)
        logit = logit.reshape(-1, self.class_num)
        y = y.reshape(-1, )
        output = (hidden.detach(), logit,)
        if y is not None:
            if is_reduction:
                loss = self.loss_reduction(logit, y)
            else:
                loss = self.loss_no_reduction(logit, y)
            output += (loss, )
        return output

import numpy as np
class LSTM_text(nn.Module):
    def __init__(self, args, embed_weight: np.ndarray):
        super(LSTM_text, self).__init__()
        self.args = args
        C = args.class_num
        # one for padded word
        V = embed_weight.shape[0]
        D = embed_weight.shape[1]

        self.embed = nn.Embedding(V, D, padding_idx=0).from_pretrained(torch.FloatTensor(embed_weight), freeze=False)

        self.lstm = nn.LSTM(D, hidden_size=args.hidden_dim//2, num_layers=args.rnn_layers,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(args.cnn_drop_out)

        self.head_count = getattr(self.args, "head_count", 1)
        # self.final_prediction = nn.Linear(self.args.hidden_size, C*self.head_count)
        # self.fc2 = nn.Linear(len(Ks) * Co, self.args.hidden_size)

        self.final_prediction = nn.Linear(args.hidden_dim, C)
        self.class_num = args.class_num

        self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.loss_reduction = nn.CrossEntropyLoss(ignore_index=-1)

        self.pad_idx = 0
        self.is_ner = getattr(args, "is_ner", False)
        self.pooling_fn = "mean" if getattr(args, "pool_fn", "mean") == "mean" else "max"


    def forward(self, x, y=None, is_reduction=True, **kwargs):
        x_length = torch.sum(x != self.pad_idx, dim=1)
        embed_x = self.embed(x)
        sorted_seq_len, permIdx = x_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_embed_x = embed_x[permIdx]
        packed_words = pack_padded_sequence(sorted_embed_x, sorted_seq_len.cpu(), True)
        lstm_out, (hidden, cell) = self.lstm(packed_words, None)
        if self.is_ner:
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)
            feature_out = self.dropout(lstm_out)
        else:
            hidden = hidden.transpose(1, 0)
            # take the last layer hidden state.
            hidden = hidden[:, -2:].reshape(hidden.shape[0], -1)
            feature_out = self.dropout(hidden)
        # reorder the output
        feature_out = feature_out[recover_idx]

        logit = self.final_prediction(feature_out)
        logit = logit.reshape(-1, self.class_num)
        y = y.reshape(-1, )
        output = (feature_out.detach(), logit,)
        if y is not None:
            if is_reduction:
                loss = self.loss_reduction(logit, y)
            else:
                loss = self.loss_no_reduction(logit, y)
            output += (loss, )
        return output

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.2))
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Transformer_Text(nn.Module):
    def __init__(self, args):
        super(Transformer_Text, self).__init__()
        config = AutoConfig.from_pretrained(
            args.transformer_model_name, output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(args.transformer_model_name, config=config)
        self.head_count = getattr(args, "head_count", 1)
        self.class_num = args.class_num
        self.final_prediction = ClassificationHead(config, num_labels=self.class_num * self.head_count)

        self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.loss_reduction = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None, is_reduction=True, **kwargs):
        device = x.device
        # lazy make the attention mask
        attention_mask = torch.where(x == 1, torch.zeros_like(x), torch.ones_like(x)).float().to(device)
        outputs = self.model(input_ids=x, attention_mask=attention_mask)
        # take <s> token (equiv. to [CLS])
        hidden = outputs[0][:, 0, :]
        logit = self.final_prediction(hidden)
        logit = logit.reshape(-1, self.class_num)
        y = y.reshape(-1, )
        output = (hidden, logit,)
        if y is not None:
            if is_reduction:
                loss = self.loss_reduction(logit, y)
            else:
                loss = self.loss_no_reduction(logit, y)
            output += (loss,)
        return output