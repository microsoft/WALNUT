import torch
import pytorch_lightning as pl
import numpy as np
from Util.util import evaluation
from Dataset.dataset import NoiseDataset
from transformers.trainer import get_parameter_names
from transformers import AdamW, get_scheduler
import math
from transformers.trainer_utils import SchedulerType
from typing import List, Any
import numpy as np

class SimpleTrainer(pl.LightningModule):
    def __init__(self, hparams, elmo_data= None):
        super(SimpleTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        self.elmo_map = None
        self.elmo_weight = None
        if elmo_data is not None:
            # 0 for word padding
            self.elmo_map = {i[0]: index + 1 for index, i in enumerate(elmo_data)}
            self.elmo_weight = np.stack([np.zeros((elmo_data[0][1].size)), *[i[1] for i in elmo_data]], axis=0)




    def forward(self, **inputs):
        return self.basic_model(**inputs)

    def _eval_end(self, outputs) -> tuple:
        try:
            loss = np.mean([[i[j].cpu().item() for j in i.keys() if "loss" in j] for i in outputs])
        except:
            loss = -1
        logits = np.concatenate([x["logits"].cpu().numpy() for x in outputs], axis=0)
        out_label_ids = np.concatenate([x["target"].cpu().numpy() for x in outputs], axis=0)
        # compute the task dependent metrics
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        results = {"loss":loss, **evaluation(logits, out_label_ids)}
        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        for key, value in logs.items():
            if type(value) is str:
                self.logger.experiment.add_text("Val/" + key,
                                                value, self.current_epoch)
            else:
                self.logger.experiment.add_scalar("Val/" + key,
                                              value, self.current_epoch)
        self.log("val_acc", logs['acc'])
        return {"val_loss": logs['loss'], "log": logs, 'val_acc': logs['acc']}

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._eval_end(outputs)
        logs = ret["log"]
        for key, value in logs.items():
            if type(value) is str:
                self.logger.experiment.add_text("Test/" + key,
                                                  value, self.current_epoch)
            else:
                self.logger.experiment.add_scalar("Test/" + key,
                                              value, self.current_epoch)
        return {"avg_test_loss": 0, "log": logs}

    def configure_optimizers(self):
        if self.hparams.is_transformer:
            # we did not utilize the gradient accumulation steps
            max_steps = math.ceil(len(self.train_dataloader()) * self.hparams.epochs)
            return self.create_roberta_optimizer_and_scheduler(max_steps)
        else:
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.basic_model.parameters()))
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.basic_model.parameters()),
                                        lr=self.hparams.basic_lr_rate, momentum=getattr(self.hparams, "momentum", 0.9))
            return [optimizer]

    def create_roberta_optimizer_and_scheduler(self, num_training_steps):
        optimizer = self.create_roberta_optimizer()
        lr_scheduler = self.create_roberta_scheduler(optimizer, num_training_steps)
        return [optimizer], [lr_scheduler]

    def create_roberta_optimizer(self):
        decay_parameters = get_parameter_names(self.basic_model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.basic_model.named_parameters() if n in decay_parameters],
                # "weight_decay": self.hparams.weight_decay,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.basic_model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        # We need to set the default value here.
        optimizer_kwargs = {
                # "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
                "betas": (0.9, 0.999),
                "eps": getattr(self.hparams, "adam_eps", 1e-08),}
        # default is 2e-05
        optimizer_kwargs["lr"] = self.hparams.basic_lr_rate
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        return optimizer

    def create_roberta_scheduler(self, optimizer, num_training_steps):
        # warmup_steps = (
        #     self.hparams.warmup_steps
        #     if self.hparams.warmup_steps > 0
        #     else math.ceil(num_training_steps * self.hparams.warmup_ratio)
        # )

        # default setting
        warmup_steps = 0

        lr_scheduler = get_scheduler(
            SchedulerType.LINEAR,
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def get_loader(self, train_type):
        dataset = NoiseDataset(self.hparams, train_type, elmo_map=self.elmo_map)
        if train_type == "train":
            shuffle = True
            batch_size = self.hparams.train_batch_size
        else:
            shuffle = False
            batch_size = self.hparams.eval_batch_size
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle)
        return dataloader

    def train_dataloader(self):
        dataloader = self.get_loader(train_type="train")
        return dataloader

    def val_dataloader(self):
        dataloader = self.get_loader(train_type="val")
        return dataloader

    def test_dataloader(self):
        dataloader = self.get_loader(train_type="test")
        return dataloader
