import os
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

from .hdf5_dataset import HDF5Dataset, TempVelDataset
from .metrics import compute_metrics, write_metrics
from .losses import LpLoss
from .plt_util import plt_temp, plt_iter_mae
from .heatflux import heatflux
from .dist_utils import local_rank, is_leader_process
from .downsample import downsample_domain

from torch.cuda import nvtx
import time

t_bulk_map = {"wall_super_heat": 58, "subcooled": 50}


class TempTrainer:
    def __init__(
        self,
        model,
        future_window,
        push_forward_steps,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        val_variable,
        writer,
        cfg,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_variable = val_variable
        self.writer = writer
        self.cfg = cfg
        self.loss = LpLoss(d=2, reduce_dims=[0, 1])

        self.push_forward_steps = push_forward_steps
        self.future_window = future_window
        self.local_rank = local_rank()

    def train(self, max_epochs, *args, **kwargs):
        for epoch in range(max_epochs):
            print("epoch ", epoch)
            self.train_step(epoch)
            self.val_step(epoch)
            # test each epoch
            val_dataset = self.val_dataloader.dataset.datasets[0]
            self.test(val_dataset)

    def _forward_int(self, coords, temp, vel):
        input = torch.cat((temp, vel), dim=1)
        if self.cfg.train.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input)
        return pred

    def push_forward_trick(self, coords, temp, vel):
        if self.cfg.train.noise:
            temp += torch.empty_like(temp).normal_(0, 0.01)
            vel += torch.empty_like(vel).normal_(0, 0.01)
        pred = self._forward_int(coords, temp, vel)
        return pred

    def train_step(self, epoch):
        self.model.train()

        for iter, (coords, temp, vel, label) in enumerate(self.train_dataloader):
            coords = coords.to(self.local_rank).float()
            temp = temp.to(self.local_rank).float()
            vel = vel.to(self.local_rank).float()
            label = label.to(self.local_rank).float()
            coords, temp, vel, label = downsample_domain(
                self.cfg.train.downsample_factor, coords, temp, vel, label
            )

            pred = self.push_forward_trick(coords, temp, vel)

            print(pred.size(), label.size())

            loss = self.loss(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()

            mse_loss = F.mse_loss(pred, label).detach()
            print(f"train loss: {loss}, mse: {mse_loss}")
            global_iter = epoch * len(self.train_dataloader) + iter
            write_metrics(pred, label, global_iter, "Train", self.writer)
            del temp, vel, label

    def val_step(self, epoch):
        self.model.eval()
        for iter, (coords, temp, vel, label) in enumerate(self.val_dataloader):
            coords = coords.to(self.local_rank).float()
            temp = temp.to(self.local_rank).float()
            vel = vel.to(self.local_rank).float()
            label = label.to(self.local_rank).float()
            with torch.no_grad():
                pred = self._forward_int(coords, temp, vel)
                temp_loss = F.mse_loss(pred, label)
                loss = temp_loss
            print(f"val loss: {loss}")
            global_iter = epoch * len(self.val_dataloader) + iter
            write_metrics(pred, label, global_iter, "Val", self.writer)
            del temp, vel, label

    def test(self, dataset, max_timestep=200):
        if is_leader_process():
            self.model.eval()
            temps = []
            labels = []
            time_lim = min(len(dataset), max_timestep)

            start = time.time()
            for timestep in range(0, time_lim, self.future_window):
                coords, temp, vel, label = dataset[timestep]
                coords = coords.to(self.local_rank).float().unsqueeze(0)
                temp = temp.to(self.local_rank).float().unsqueeze(0)
                vel = vel.to(self.local_rank).float().unsqueeze(0)
                label = label.to(self.local_rank).float()
                with torch.no_grad():
                    pred = self._forward_int(coords, temp, vel)
                    temp = F.hardtanh(pred.squeeze(0), -1, 1)
                    dataset.write_temp(temp, timestep)
                    temps.append(temp.detach().cpu())
                    labels.append(label.detach().cpu())
            dur = time.time() - start
            print(f"rollout time {dur} (s)")

            temps = torch.cat(temps, dim=0)
            labels = torch.cat(labels, dim=0)
            dfun = dataset.get_dfun()[: temps.size(0)]

            print(temps.max(), temps.min())
            print(labels.max(), labels.min())

            metrics = compute_metrics(temps, labels, dfun)
            print(metrics)

            # xgrid = dataset.get_x().permute((2, 0, 1))
            # print(heatflux(temps, dfun, self.val_variable, xgrid, dataset.get_dy()))
            # print(heatflux(labels, dfun, self.val_variable, xgrid, dataset.get_dy()))

            plt_temp(temps, labels, self.model.__class__.__name__)
            plt_iter_mae(temps, labels)

            dataset.reset()

            return metrics
