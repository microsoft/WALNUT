from Trainer.simple_trainer import SimpleTrainer
from Model.BasicEncoder import CNN_Text, Transformer_Text, LSTM_text
import torch
import numpy as np
from argparse import ArgumentParser
import torch.nn as nn
from transformers import AdamW, get_scheduler
from Util.util import evaluation
class BaselineTrainer(SimpleTrainer):
    def __init__(self, hparams, elmo_data=None):
        super(BaselineTrainer, self).__init__(hparams, elmo_data)
        if getattr(hparams, 'is_transformer', False) is False:
            self.basic_model = LSTM_text(hparams, self.elmo_weight)
            model_parameters = filter(lambda p: p.requires_grad, self.basic_model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("THERE ARE {} Params".format(params
                                               ))
        else:
            self.basic_model = Transformer_Text(hparams)
            self.automatic_optimization = False

        self.clean_ratio = hparams.clean_ratio

    def validation_step(self, *args, **kwargs):
        batch = args[0]
        if type(batch) is dict:
            input_ids = batch['input_ids']
            label = batch['label']
        else:
            input_ids, _, label, _ = batch

        _, logits, loss = self.basic_model(input_ids, label)
        return {"logits": logits, "loss": loss, "target": label}

    def training_step(self, *args, **kwargs):
        batch = args[0]
        if type(batch) is dict:
            input_ids = batch['input_ids']
            label = batch['label']
        else:
            input_ids, _, label, _ = batch
        _, logits, loss = self.basic_model(input_ids, label)
        tensorboard_logs = {"loss": loss}
        if self.hparams.is_transformer:
            # manually do the backpropagation for roberta model
            lr_sch = self.lr_schedulers()
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            lr_sch.step()
            tensorboard_logs['lr'] = optimizer.defaults.get("lr", 0)

        return {"loss": loss, "log": tensorboard_logs, "logits":logits, 'target':label}

    def training_epoch_end(self, outputs):
        loss = np.mean([i['loss'].item() for i in outputs])
        logits = np.concatenate([x["logits"].detach().cpu().numpy() for x in outputs], axis=0)
        out_label_ids = np.concatenate([x["target"].cpu().numpy() for x in outputs], axis=0)
        result = evaluation(logits, out_label_ids)
        self.logger.experiment.add_scalar("Train/" + "Loss",
                                          loss, self.current_epoch)
        for key, value in result.items():
            if type(value) is str:
                self.logger.experiment.add_text("Train/Epoch-" + key,
                                                  value, self.current_epoch)
            else:
                self.logger.experiment.add_scalar("Train/Epoch-" + key,
                                                  value, self.current_epoch)


    @staticmethod
    def add_model_specific_args(parent_parser):
        #TODO: Other Hyper-parameter in the baseline methods? The augmented data selection?
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser