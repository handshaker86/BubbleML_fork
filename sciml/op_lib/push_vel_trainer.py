import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .dist_utils import local_rank
from .downsample import downsample_domain
from .losses import LpLoss
from .metrics import compute_metrics, write_metrics
from .plt_util import plt_temp, plt_iter_mae, plt_vel


class PushVelTrainer:
    def __init__(
        self,
        model,
        future_window,
        max_push_forward_steps,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        val_variable,
        writer,
        cfg,
        result_save_path,
        train_max_temp,
        train_max_vel,
        prediction_save_path=None,
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

        self.max_push_forward_steps = max_push_forward_steps
        self.future_window = future_window
        self.use_coords = cfg.train.use_coords
        self.result_save_path = result_save_path
        self.train_max_temp = train_max_temp
        self.train_max_vel = train_max_vel
        self.prediction_save_path = prediction_save_path

    def save_checkpoint(self, log_dir, dataset_name):
        timestamp = int(time.time())
        if self.cfg.distributed:
            model_name = self.model.module.__class__.__name__
        else:
            model_name = self.model.__class__.__name__
        ckpt_file = f"{model_name}_{self.cfg.torch_dataset_name}_{self.cfg.train.max_epochs}_{timestamp}.pt"
        ckpt_root = f"{log_dir}/{dataset_name}/training_model_ckpt"
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)
        ckpt_path = f"{ckpt_root}/{ckpt_file}"
        print(f"saving model to {ckpt_path}")
        if self.cfg.distributed:
            torch.save(self.model.module.state_dict(), f"{ckpt_path}")
        else:
            torch.save(self.model.state_dict(), f"{ckpt_path}")

    def push_forward_prob(self, epoch, max_epochs):
        """Probabilistic push-forward steps: rare early, frequent late in training."""
        cur_iter = epoch * len(self.train_dataloader)
        tot_iter = max_epochs * len(self.train_dataloader)
        frac = cur_iter / tot_iter
        if np.random.uniform() > frac:
            return 1
        else:
            return self.max_push_forward_steps

    def train(self, max_epochs, log_dir, dataset_name):
        for epoch in range(max_epochs):
            print("epoch", epoch)
            self.train_step(epoch, max_epochs)

    def _forward_int(self, coords, temp, vel, dfun):
        inp = torch.cat((temp, vel, dfun), dim=1)
        if self.use_coords:
            inp = torch.cat((coords, inp), dim=1)
        pred = self.model(inp)
        temp_pred = pred[:, : self.future_window]
        vel_pred = pred[:, self.future_window :]
        return temp_pred, vel_pred

    def _index_push(self, idx, coords, temp, vel, dfun):
        """Select channels for push-forward step idx."""
        return (coords[:, idx], temp[:, idx], vel[:, idx], dfun[:, idx])

    def _index_dfun(self, idx, dfun):
        return dfun[:, idx]

    def push_forward_trick(self, coords, temp, vel, dfun, push_forward_steps):
        coords_input, temp_input, vel_input, dfun_input = self._index_push(
            0, coords, temp, vel, dfun
        )
        assert self.future_window == temp_input.size(
            1
        ), "push-forward expects history size to match future"
        coords_input, temp_input, vel_input, dfun_input = downsample_domain(
            self.cfg.train.downsample_factor,
            coords_input,
            temp_input,
            vel_input,
            dfun_input,
        )
        with torch.no_grad():
            for idx in range(push_forward_steps - 1):
                temp_input, vel_input = self._forward_int(
                    coords_input, temp_input, vel_input, dfun_input
                )
                dfun_input = self._index_dfun(idx + 1, dfun)
                dfun_input = downsample_domain(
                    self.cfg.train.downsample_factor, dfun_input
                )[0]
        if self.cfg.train.noise and push_forward_steps == 1:
            temp_input += torch.empty_like(temp_input).normal_(0, 0.01)
            vel_input += torch.empty_like(vel_input).normal_(0, 0.01)
        temp_pred, vel_pred = self._forward_int(
            coords_input, temp_input, vel_input, dfun_input
        )
        return temp_pred, vel_pred

    def train_step(self, epoch, max_epochs):
        self.model.train()
        for iter, (coords, temp, vel, dfun, temp_label, vel_label) in enumerate(
            self.train_dataloader
        ):
            coords = coords.to(local_rank()).float()
            temp = temp.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()

            push_forward_steps = self.push_forward_prob(epoch, max_epochs)

            temp_pred, vel_pred = self.push_forward_trick(
                coords, temp, vel, dfun, push_forward_steps
            )

            idx = push_forward_steps - 1
            temp_label = temp_label[:, idx].to(local_rank()).float()
            vel_label = vel_label[:, idx].to(local_rank()).float()

            temp_label, vel_label = downsample_domain(
                self.cfg.train.downsample_factor, temp_label, vel_label
            )

            temp_loss = F.mse_loss(temp_pred, temp_label)
            vel_loss = F.mse_loss(vel_pred, vel_label)
            loss = (temp_loss + vel_loss) / 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            print(f"train loss: {loss}")
            global_iter = epoch * len(self.train_dataloader) + iter
            write_metrics(temp_pred, temp_label, global_iter, "TrainTemp", self.writer)
            write_metrics(vel_pred, vel_label, global_iter, "TrainVel", self.writer)
            del temp, vel, temp_label, vel_label

    def val_step(self, epoch):
        self.model.eval()
        for iter, (coords, temp, vel, dfun, temp_label, vel_label) in enumerate(
            self.val_dataloader
        ):
            coords = coords.to(local_rank()).float()
            temp = temp.to(local_rank()).float()
            vel = vel.to(local_rank()).float()
            dfun = dfun.to(local_rank()).float()

            # val doesn't apply push-forward
            temp_label = temp_label[:, 0].to(local_rank()).float()
            vel_label = vel_label[:, 0].to(local_rank()).float()

            with torch.no_grad():
                temp_pred, vel_pred = self._forward_int(
                    coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0]
                )
                temp_loss = F.mse_loss(temp_pred, temp_label)
                vel_loss = F.mse_loss(vel_pred, vel_label)
                loss = (temp_loss + vel_loss) / 2
            print(f"val loss: {loss}")
            global_iter = epoch * len(self.val_dataloader) + iter
            write_metrics(temp_pred, temp_label, global_iter, "ValTemp", self.writer)
            write_metrics(vel_pred, vel_label, global_iter, "ValVel", self.writer)
            del temp, vel, temp_label, vel_label

    @staticmethod
    def _inverse_transform(data, scale):
        """Inverse normalization."""
        return data * scale

    def test(self, dataset, max_time_limit=200, warmup_runs=5, average_runs=10):
        self.model.eval()
        print(f"Warming up GPU...")
        for _ in range(warmup_runs):
            coords, temp, vel, dfun, temp_label, vel_label = dataset[0]
            coords = coords.to(local_rank()).float().unsqueeze(0)
            temp = temp.to(local_rank()).float().unsqueeze(0)
            vel = vel.to(local_rank()).float().unsqueeze(0)
            dfun = dfun.to(local_rank()).float().unsqueeze(0)
            temp_pred, vel_pred = self._forward_int(
                coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0]
            )
        torch.cuda.synchronize()
        print("GPU is ready.")

        num_runs = max(1, average_runs)
        all_run_times = []
        time_limit = min(max_time_limit, len(dataset))
        temp_scale = self.train_max_temp
        vel_scale = self.train_max_vel

        for run_idx in range(num_runs):
            dataset.reset()
            self.model.eval()
            temps = []
            temps_labels = []
            vels = []
            vels_labels = []
            total_prediction_time = 0.0

            for timestep in range(0, time_limit, self.future_window):
                coords, temp, vel, dfun, temp_label, vel_label = dataset[timestep]
                coords = coords.to(local_rank()).float().unsqueeze(0)
                temp = temp.to(local_rank()).float().unsqueeze(0)
                vel = vel.to(local_rank()).float().unsqueeze(0)
                dfun = dfun.to(local_rank()).float().unsqueeze(0)
                temp_label = temp_label[0].to(local_rank()).float()
                vel_label = vel_label[0].to(local_rank()).float()
                temp_label = self._inverse_transform(temp_label, temp_scale)
                vel_label = self._inverse_transform(vel_label, vel_scale)

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    temp_pred, vel_pred = self._forward_int(
                        coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0]
                    )
                    temp_pred = temp_pred.squeeze(0)
                    vel_pred = vel_pred.squeeze(0)
                    dataset.write_temp(temp_pred, timestep)
                    dataset.write_vel(vel_pred, timestep)
                    temp_pred = self._inverse_transform(temp_pred, temp_scale)
                    vel_pred = self._inverse_transform(vel_pred, vel_scale)
                    temps.append(temp_pred.detach().cpu())
                    temps_labels.append(temp_label.detach().cpu())
                    vels.append(vel_pred.detach().cpu())
                    vels_labels.append(vel_label.detach().cpu())

                torch.cuda.synchronize()
                end_time = time.perf_counter()
                total_prediction_time += end_time - start_time

            all_run_times.append(total_prediction_time)

        if not all_run_times:
            total_prediction_time = 0.0
        else:
            total_prediction_time = sum(all_run_times) / len(all_run_times)

        frame_prediction_time = total_prediction_time / time_limit

        temps = torch.cat(temps, dim=0)
        temps_labels = torch.cat(temps_labels, dim=0)
        vels = torch.cat(vels, dim=0)
        vels_labels = torch.cat(vels_labels, dim=0)
        dfun = dataset.get_dfun()[: temps.size(0)]

        velx_preds = vels[0::2]
        velx_labels = vels_labels[0::2]
        vely_preds = vels[1::2]
        vely_labels = vels_labels[1::2]
        temp_metrics = compute_metrics(temps, temps_labels, dfun)
        print("TEMP METRICS")
        print(temp_metrics)
        velx_metrics = compute_metrics(velx_preds, velx_labels, dfun)
        print("VELX METRICS")
        print(velx_metrics)
        vely_metrics = compute_metrics(vely_preds, vely_labels, dfun)
        print("VELY METRICS")
        print(vely_metrics)
        vel_preds = torch.stack((velx_preds, vely_preds), dim=1)
        vel_labels = torch.stack((velx_labels, vely_labels), dim=1)
        vel_metrics = compute_metrics(vel_preds, vel_labels, dfun)
        print("VELOCITY METRICS")
        print(vel_metrics)

        os.makedirs(self.result_save_path, exist_ok=True)
        with open(self.result_save_path / "metrics.txt", "w") as f:
            f.write(f"Temp metrics: {temp_metrics}\n")
            f.write(f"Velx metrics: {velx_metrics}\n")
            f.write(f"Vely metrics: {vely_metrics}\n")
            f.write(f"Vel metrics: {vel_metrics}\n")

        with open(self.result_save_path / "loss.txt", "w") as f:
            f.write(f"RMSE: {vel_metrics.rmse}\n")
            f.write(f"Rel L2: {vel_metrics.rel_l2}\n")
            f.write(f"Rel Vorticity: {vel_metrics.rel_vort}\n")
            f.write(f"Div Error: {vel_metrics.div_error}\n")
            f.write(f"Spectral Error: {vel_metrics.spectral_error}\n")

        with open(self.result_save_path / "predict_time.txt", "w") as f:
            f.write(f"Total prediction time: {total_prediction_time}\n")
            f.write(f"Frame prediction time: {frame_prediction_time}\n")

        if self.prediction_save_path is not None:
            pred_velx = velx_preds.numpy()
            pred_vely = vely_preds.numpy()
            assert pred_velx.ndim == 3 and pred_vely.ndim == 3
            n_frames = pred_velx.shape[0]
            pred_dict = {
                f"frame_{i}": np.stack([pred_velx[i], pred_vely[i]], axis=0)
                for i in range(n_frames)
            }
            np.savez(self.prediction_save_path, **pred_dict)
            print(
                f"Prediction saved: {self.prediction_save_path} "
                f"frames={n_frames} (u,v) each (2,h,w)"
            )

        model_name = self.model.__class__.__name__
        plt_iter_mae(temps, temps_labels)
        plt_temp(temps, temps_labels, model_name)

        def mag(velx, vely):
            return torch.sqrt(velx**2 + vely**2)

        mag_preds = mag(velx_preds, vely_preds)
        mag_labels = mag(velx_labels, vely_labels)

        plt_vel(
            mag_preds,
            mag_labels,
            velx_preds,
            velx_labels,
            vely_preds,
            vely_labels,
            model_name,
        )

        dataset.reset()

    def benchmark_predict_100_frames(
        self,
        dataset,
        num_frames=100,
        warmup_runs=5,
        average_runs=10,
    ):
        """Benchmark prediction time for num_frames; returns mean, std, per-frame time."""
        self.model.eval()
        time_limit = min(num_frames, len(dataset))
        if time_limit < self.future_window:
            time_limit = min(self.future_window, len(dataset))
        actual_frames = (time_limit // self.future_window) * self.future_window

        print(f"Warming up GPU ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            coords, temp, vel, dfun, temp_label, vel_label = dataset[0]
            coords = coords.to(local_rank()).float().unsqueeze(0)
            temp = temp.to(local_rank()).float().unsqueeze(0)
            vel = vel.to(local_rank()).float().unsqueeze(0)
            dfun = dfun.to(local_rank()).float().unsqueeze(0)
            temp_pred, vel_pred = self._forward_int(
                coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0]
            )
        torch.cuda.synchronize()
        print("GPU ready.")

        temp_scale = self.train_max_temp
        vel_scale = self.train_max_vel
        all_run_times = []
        num_runs = max(1, average_runs)

        for run_idx in range(num_runs):
            dataset.reset()
            self.model.eval()
            temps = []
            temps_labels = []
            vels = []
            vels_labels = []
            total_prediction_time = 0.0

            for timestep in range(0, time_limit, self.future_window):
                coords, temp, vel, dfun, temp_label, vel_label = dataset[timestep]
                coords = coords.to(local_rank()).float().unsqueeze(0)
                temp = temp.to(local_rank()).float().unsqueeze(0)
                vel = vel.to(local_rank()).float().unsqueeze(0)
                dfun = dfun.to(local_rank()).float().unsqueeze(0)
                temp_label = temp_label[0].to(local_rank()).float()
                vel_label = vel_label[0].to(local_rank()).float()
                temp_label = self._inverse_transform(temp_label, temp_scale)
                vel_label = self._inverse_transform(vel_label, vel_scale)

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    temp_pred, vel_pred = self._forward_int(
                        coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0]
                    )
                    temp_pred = temp_pred.squeeze(0)
                    vel_pred = vel_pred.squeeze(0)
                    dataset.write_temp(temp_pred, timestep)
                    dataset.write_vel(vel_pred, timestep)
                    temp_pred = self._inverse_transform(temp_pred, temp_scale)
                    vel_pred = self._inverse_transform(vel_pred, vel_scale)
                    temps.append(temp_pred.detach().cpu())
                    temps_labels.append(temp_label.detach().cpu())
                    vels.append(vel_pred.detach().cpu())
                    vels_labels.append(vel_label.detach().cpu())

                torch.cuda.synchronize()
                end_time = time.perf_counter()
                total_prediction_time += end_time - start_time

            all_run_times.append(total_prediction_time)

        mean_time = sum(all_run_times) / len(all_run_times)
        variance = sum((t - mean_time) ** 2 for t in all_run_times) / len(all_run_times)
        std_time = variance**0.5
        frame_time = mean_time / actual_frames

        print(f"Benchmark: {actual_frames} frames, {num_runs} runs")
        print(f"  Total time: {mean_time:.4f} ± {std_time:.4f} s")
        print(f"  Per-frame:  {frame_time:.6f} s")
        dataset.reset()
        return mean_time, std_time, frame_time, actual_frames
