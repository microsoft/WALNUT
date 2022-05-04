from Trainer.simple_trainer import SimpleTrainer
from Model.BasicEncoder import CNN_Text, Transformer_Text, LSTM_text
from Model.MetaWeight import FullWeightModel, MetaNet
from Trainer.k_step_meta import step_hmlc_K
from Trainer.meta_process import step_l2w_group_net, step_l2r
import torch
import numpy as np
from argparse import ArgumentParser
from transformers import get_scheduler
from Dataset.dataset import NoiseDataset
from torch.utils.data import Subset

class DummyScheduler(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])

        return lrs

    def step(self, epoch=None):
        pass

class MetaTrainer(SimpleTrainer):
    def __init__(self, hparams):
        super(MetaTrainer, self).__init__(hparams)
        print(hparams)
        self.is_transformer = getattr(hparams, 'is_transformer', False)
        if self.is_transformer is False:
            self.basic_model = LSTM_text(hparams)
        else:
            self.basic_model = Transformer_Text(hparams)
        self.meta_weight = MetaNet(hparams)
        self.dw_prev = [0 for p in self.meta_weight.parameters() if p.requires_grad]
        self.automatic_optimization = False

    def validation_step(self, *args, **kwargs):
        batch = args[0]

        if type(batch) is dict:
            x_clean = batch['input_ids']
            y_clean = batch['label']
        else:
            x_clean, _, y_clean, _ = batch
        _, logits, loss = self.basic_model(x_clean, y_clean)
        return {"logits":logits, "loss":loss, "target":y_clean}

    def training_step(self, batch, batch_idx, optimizer_idx):
        basic_batch = batch['base_loader']
        meta_batch = batch['meta_loader']
        x_weak, _, y_weak, y_weak_2d = basic_batch
        x_clean, _, y_clean, _ = meta_batch
        (basic_optimizer, meta_optimizer) = self.optimizers()
        (basic_scheduler, meta_scheduler) = self.lr_schedulers()
        # pre-train the meta model and basic encoder iteratively

        if self.current_epoch < (self.hparams.pre_train_basic_epochs + self.hparams.pre_train_meta_epochs)\
            and self.meta_weight.weight_output_dim == 1:
            # the pre-training is only for instance level weight
            loss = self.pre_step(x_clean, y_clean, basic_optimizer, meta_optimizer)
            return {"loss_pre": loss}
        else:
            # 1. My version 2. Guoqing version 3. Guoqing aaai
            if self.hparams.meta_fn == "hmlc":
                meta_fn = step_hmlc_K
            elif self.hparams.meta_fn == "mwss":
                meta_fn = step_l2w_group_net
            elif self.hparams.meta_fn == "l2r":
                meta_fn = step_l2r
            else:
                raise NotImplementedError

            loss_val, loss_s, log_loss_s, loss_train_clean, loss_final, instance_weight = meta_fn(meta_trainer=self,
                                                                                                  main_net=self.basic_model, main_opt=basic_optimizer,
                                                                                                  meta_net=self.meta_weight, meta_opt=meta_optimizer,
                                                                                                  val_clean_data={"x": x_clean, "y": y_clean},
                                                                                                  train_weak_data={"x": x_weak, 'y': y_weak}, main_scheduler=basic_scheduler)

            tensorboard_logs = {"loss":loss_final, "loss_s":loss_s, "mean_loss_s":log_loss_s, 'loss_train_clean':loss_train_clean, 'loss_val':loss_val}
        # if self.is_transformer:
        basic_scheduler.step()
        return {"loss": loss_final, "log":tensorboard_logs, "instance_weight":instance_weight}


    def training_epoch_end(self, outputs):
        # outputs = outputs[0]

        if type(outputs[0]) is list:
            outputs = outputs[0]
        loss = np.mean([i['loss'].item() for i in outputs])
        loss_s = np.mean([i['log']['loss_s'].item() for i in outputs])
        mean_loss_s = np.mean([i['log']['mean_loss_s'].item() for i in outputs])
        loss_train_clean = np.mean([i['log']['loss_train_clean'].item() for i in outputs])
        loss_g = np.mean([i['log']['loss_val'].item() for i in outputs])


        self.logger.experiment.add_scalar("Train/" + "Loss",
                                          loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train/" + "Loss_s",
                                          loss_s, self.current_epoch)
        self.logger.experiment.add_scalar("Train/" + "Mean_loss_s",
                                          mean_loss_s, self.current_epoch)
        self.logger.experiment.add_scalar("Train/" + "loss_val",
                                          loss_g, self.current_epoch)
        self.logger.experiment.add_scalar("Train/" + "Loss_train_clean",
                                          loss_train_clean, self.current_epoch)
        if "instance_weight" in outputs[0] and outputs[0]['instance_weight'] is not None:
            try:
                instance_weight = torch.cat([i['instance_weight'].detach() for i in outputs], dim=0)
                # update the dropout rate of the augmented samples.
                instWeightNorm2 = instance_weight.norm(2)
                if getattr(self.hparams, "is_scheduler_noise", False):
                    # increase the dropout if the module did not learn much knowledge
                    new_p = self.drop_out_scheduler.step(hyper_value=(1 - self.augment_dropout.p),
                                             metrics=instWeightNorm2,
                                             epoch=self.current_epoch)
                    self.augment_dropout.p = 1 - new_p
                    self.logger.experiment.add_scalar("Train/AugDropOut", 1-new_p, self.current_epoch)
                self.logger.experiment.add_scalar("Train/InstWeightNorm2", instWeightNorm2, self.current_epoch)
                # instance_weight = np.concatenate([i['instance_weight'].detach().cpu().numpy() for i in outputs], axis=0)
                instance_weight_np = instance_weight.cpu().numpy()
                self.logger.experiment.add_histogram("Train/InstanceWeight", instance_weight_np, self.current_epoch)
            except:
                pass

    def train_dataloader(self):
        # multiple dataloader and a special one for validation
        train_loader = self.get_loader("train")
        train_meta_loader = self.get_loader("train", is_meta=True)
        return {"base_loader": train_loader, "meta_loader": train_meta_loader}

    def get_loader(self, train_type, is_meta=False):
        dataset_cls = NoiseDataset
        if is_meta:
            # utilize the validation dataset as the meta-dataset
            dataset = dataset_cls(self.hparams, "val")
        else:
            dataset = dataset_cls(self.hparams, train_type)

        if train_type == "train":
            shuffle = True
            batch_size = self.hparams.train_batch_size
        else:
            shuffle = False
            batch_size = self.hparams.eval_batch_size
        # without drop_last will cause the spikes in the training loss
        # pls check https://stacko  verflow.com/questions/47824598/why-does-my-training-loss-have-regular-spikes
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle)
        return dataloader

    def configure_optimizers(self):
        # TODO: check the scheduler
        if self.hparams.meta_fn == "hmlc" or self.hparams.meta_fn == "l2r":
            Optimizer_basic = torch.optim.SGD
        else:
            Optimizer_basic = torch.optim.AdamW
        basic_optimizer = Optimizer_basic(
            filter(lambda p: p.requires_grad, self.basic_model.parameters()),
            lr=self.hparams.basic_lr_rate,
            weight_decay=self.hparams.basic_weight_decay
        )
        meta_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.meta_weight.parameters()),
            lr=self.hparams.meta_lr_rate,
            weight_decay=self.hparams.meta_weight_decay
        )
        if self.hparams.is_transformer:
            # ATTENTION: Linear Scheduler but the optimizer is SGD not Adamw.
            basic_scheduler = get_scheduler(
                "linear",
                basic_optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams.epochs * len(self.train_dataloader()['base_loader'].dataset),
            )
            # meta module did not utilize the lr scheduler
            meta_scheduler = DummyScheduler(meta_optimizer)
            return [basic_optimizer, meta_optimizer], [basic_scheduler, meta_scheduler]
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(basic_optimizer, len(self.get_loader("train", is_meta=False)))
            return [basic_optimizer, meta_optimizer], [scheduler, scheduler]



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--meta_lr_rate", default=0.01, type=float)
        parser.add_argument("--meta_weight_decay", default=0.001, type=float)
        parser.add_argument("--cls_emb_dim", default=128, type=int)
        parser.add_argument("--gw_hidden_dim", default=256, type=int)
        parser.add_argument("--is_deeper_weight", action='store_true')
        parser.add_argument("--gw_dropout", default=0.3, type=float)
        parser.add_argument("--gradient_steps", default=1, type=int, help='Number of look-ahead gradient steps for meta-gradient (default: 1)')
        parser.add_argument("--is_guoqing_method", action='store_true', help='whether utilize Guoqing s method in aaai')
        parser.add_argument("--weight_output_dim", default=1, type=int,
                            help='if > 1 will apply the logits weight， '
                                 'this should match the number of class')
        parser.add_argument("--weight_scale", default=1.0, type=float,
                            help='')
        return parser


class MetaTrainerLast(MetaTrainer):
    def __init__(self, hparams):
        super(MetaTrainer, self).__init__(hparams)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_clean, y_clean, x_clean_val, y_clean_val, x_aug, y_aug, _, _ = batch
        basic_optimizer, _ = self.optimizers()
        x = torch.cat((x_clean, x_clean_val), dim=0)
        y = torch.cat((y_clean, y_clean_val), dim=0)
        _, logits, loss = self.basic_model(x, y)
        basic_optimizer.zero_grad()
        self.manual_backward(loss)
        basic_optimizer.step()
        tensorboard_logs = {"loss": loss, "loss_s": loss, 'loss_train_clean': loss,
                            'loss_val': loss}
        return {"loss": loss, "log": tensorboard_logs}
