# 2023 (c) LINE Corporation
# Authors: Robin Scheibler
# MIT License
import datetime
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import fast_bss_eval
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf.omegaconf import open_dict
from scipy.optimize import linear_sum_assignment
from torch_ema import ExponentialMovingAverage

import sdes
import utils

log = logging.getLogger(__name__)




def normalize_batch(batch):
    mix, tgt = batch
    mean = mix.mean(dim=(1, 2), keepdim=True)
    std = mix.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
    mix = (mix - mean) / std
    if tgt is not None:
        tgt = (tgt - mean) / std
    return (mix, tgt), mean, std


def denormalize_batch(x, mean, std):
    return x * std + mean


class DiffSepModel(pl.LightningModule):
    def __init__(self, config):
        # init superclass
        super().__init__()

        self.save_hyperparameters()

        # the config and all hyperparameters are saved by hydra to the experiment dir
        self.config = config

        self.score_model = instantiate(self.config.model.score_model, _recursive_=False)

        self.valid_max_sep_batches = getattr(
            self.config.model, "valid_max_sep_batches", 1
        )
        self.t_eps = self.config.model.t_eps
        self.time_sampling_strategy = getattr(
            self.config.model, "time_sampling_strategy", "uniform"
        )
        self.init_hack = getattr(self.config.model, "init_hack", False)
        self.t_rev_init = getattr(self.config.model, "t_rev_init", 0.03)

        self.lr_warmup = getattr(config.model, "lr_warmup", None)
        self.lr_original = self.config.model.optimizer.lr

        self.train_source_order = getattr(
            self.config.model, "train_source_order", "random"
        )

        # configure the loss functions
        if self.init_hack in [5, 6, 7]:
            if "reduction" not in self.config.model.loss:
                self.loss = instantiate(self.config.model.loss, reduction="none")
            elif self.config.model.loss.reduction != "none":
                raise ValueError("Reduction should 'none' for loss with init_hack == 5")
        else:
            self.loss = instantiate(self.config.model.loss)
        self.val_losses = {}
        for name, loss_args in self.config.model.val_losses.items():
            self.val_losses[name] = instantiate(loss_args)

        # for moving average of weights
        self.ema_decay = getattr(self.config.model, "ema_decay", 0.0)
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        self.normalize_batch = normalize_batch
        self.denormalize_batch = denormalize_batch


    def forward(self, mix):
        # implement inference here
        return self.score_model(mix)

    def compute_score_loss(self, mix, target):
        # compute the samples and associated score
        # predict the score
        est = self(mix)

        # compute the MSE loss
        loss = self.loss(est, target)

        if loss.ndim == 3:
            loss = loss.mean(dim=(-2, -1))

        return loss

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):

        batch, *stats = self.normalize_batch(batch)

        mix, target = batch

        loss = self.compute_score_loss(mix, target)

        # every 10 steps, we log stuff
        cur_step = self.trainer.global_step
        self.last_step = getattr(self, "last_step", 0)
        if cur_step > self.last_step and cur_step % 10 == 0:
            self.last_step = cur_step

            # log the classification metrics
            self.logger.log_metrics(
                {"train/score_loss": loss},
                step=cur_step,
            )

        self.do_lr_warmup()

        return loss

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        self.n_batches_est_done = 0

    def validation_step(self, batch, batch_idx, dataset_i=0):
        pass
        # batch, *stats = self.normalize_batch(batch)

        # mix, target = batch

        # # validation score loss
        # loss = self.compute_score_loss(mix, target)
        # self.log("val/score_loss", loss, on_epoch=True, sync_dist=True)

        # validation separation losses
        # if self.trainer.testing or self.n_batches_est_done < self.valid_max_sep_batches:
        #     self.n_batches_est_done += 1
        #     est, *_ = self.separate(mix)

        #     est = self.denormalize_batch(est, *stats)

            # for name, loss in self.val_losses.items():
            #     self.log(name, loss(est, target), on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        pass

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataset_i=None):
        return self.validation_step(batch, batch_idx, dataset_i=dataset_i)

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        # we may have some frozen layers, so we remove these parameters
        # from the optimization
        log.info(f"set optim with {self.config.model.optimizer}")

        opt_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate(
            {**{"params": opt_params}, **self.config.model.optimizer}
        )

        if getattr(self.config.model, "scheduler", None) is not None:
            scheduler = instantiate(
                {**self.config.model.scheduler, **{"optimizer": optimizer}}
            )
        else:
            scheduler = None

        # this will be called in on_after_backward
        self.grad_clipper = instantiate(self.config.model.grad_clipper)

        if scheduler is None:
            return [optimizer]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.model.main_val_loss,
            }

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def on_after_backward(self):
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            # we still want to compute this for monitoring in tensorboard
            grad_norm = utils.grad_norm(self)
            clipped_norm = grad_norm

        # log every few iterations
        if self.trainer.global_step % 25 == 0:
            clipped_norm = min(grad_norm, clipping_threshold)

            # get the current learning reate
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]

            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self._error_loading_ema = True
            log.warn("EMA state_dict not found in checkpoint!")

    def train(self, mode=True, no_ema=False):
        res = super().train(
            mode
        )  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                # eval
                self.ema.store(self.parameters())  # store current params in EMA
                self.ema.copy_to(
                    self.parameters()
                )  # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(
                        self.parameters()
                    )  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def do_lr_warmup(self):
        if self.lr_warmup is not None and self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            optimizer = self.trainer.optimizers[0]
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr_original