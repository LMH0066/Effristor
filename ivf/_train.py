from typing import Dict, List, Union

import numpy as np
import torch
from scvi.train import TrainingPlan
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

from ivf._utils import LOSS_KEYS, DefaultMetric


class IVFTrainingPlan(TrainingPlan):
    """Training plan for the ivf model."""

    def __init__(
        self,
        module,
        metric=DefaultMetric(),
        n_epochs_warmup: Union[int, None] = None,
        checkpoint_freq: int = 20,
        lr=1e-4,
        weight_decay=1e-4,
        step_scheduler: bool = False,
        step_size_lr: int = 4,
        gamma_lr: float = 0.1,
        cosine_scheduler: bool = False,
        scheduler_max_epochs: int = 1000,
        scheduler_final_lr: float = 1e-5,
        one_cycle_scheduler: bool = False,
        one_cycle_max_lr: float = 1e-3,
        one_cycle_total_steps: int = 1000,
        one_cycle_pct_start: float = 0.1,
        gclip: float = 0,
        log_sync_dist: bool = False,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs_kl_warmup=None,
            reduce_lr_on_plateau=False,
            lr_factor=None,
            lr_patience=None,
            lr_threshold=None,
            lr_scheduler_metric=None,
            lr_min=None,
        )

        self.n_epochs_warmup = n_epochs_warmup if n_epochs_warmup is not None else 0

        self.checkpoint_freq = checkpoint_freq

        self.scheduler = None
        if step_scheduler:
            self.scheduler = StepLR
            self.scheduler_params = {"step_size": step_size_lr, "gamma": gamma_lr}
        elif cosine_scheduler:
            self.scheduler = CosineAnnealingLR
            self.scheduler_params = {
                "T_max": scheduler_max_epochs,
                "eta_min": scheduler_final_lr,
            }
        elif one_cycle_scheduler:
            self.scheduler = OneCycleLR
            self.scheduler_params = {
                "max_lr": one_cycle_max_lr,
                "total_steps": one_cycle_total_steps,
                "pct_start": one_cycle_pct_start,
            }

        self.step_size_lr = step_size_lr

        self.automatic_optimization = False
        self.iter_count = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self._epoch_keys = []

        self.use_losses = [LOSS_KEYS.FocalLoss]
        self.metric = metric.to(self.module.device)
        self.epoch_keys = ["metric"] + self.use_losses

        self.epoch_history = {"mode": [], "epoch": []}
        for key in self.epoch_keys:
            self.epoch_history[key] = []

        self.gclip = gclip

        self.sync_dist = log_sync_dist

    def configure_optimizers(self):
        """Set up optimizers."""
        optimizers = []
        schedulers = []

        optimizers.append(
            torch.optim.AdamW(
                [
                    {
                        "params": list(
                            filter(
                                lambda p: p.requires_grad,
                                self.module.parameters(),
                            )
                        ),
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                        # betas=(0.5, 0.999),
                    }
                ]
            )
        )

        if self.scheduler is not None:
            for optimizer in optimizers:
                schedulers.append(self.scheduler(optimizer, **self.scheduler_params))
            return optimizers, schedulers
        else:
            return optimizers

    @property
    def epoch_keys(self):
        """Epoch keys getter."""
        return self._epoch_keys

    @epoch_keys.setter
    def epoch_keys(self, epoch_keys: List):
        self._epoch_keys.extend(epoch_keys)

    def training_step(self, batch):
        """Training step."""
        optimizers = self.optimizers()
        for name, param in self.module.named_parameters():
            if "cope" not in name and "embedding" not in name:
                continue
            if "bias" in name:
                continue
            if param.grad is not None:
                self.log("gradient/{}".format(name), param.grad.norm().item())

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        # model update
        for optimizer in optimizers:
            self.log(
                "lr/{}".format(optimizer.__class__.__name__),
                optimizer.state_dict()["param_groups"][0]["lr"],
            )
            optimizer.zero_grad()

        _, losses = self.module.forward(batch)

        for key in self.use_losses:
            self.manual_backward(losses[key])
        if self.gclip > 0:
            torch.nn.utils.clip_grad_value_(self.module.parameters(), self.gclip)
        for optimizer in optimizers:
            optimizer.step()

        results = {}
        for key in self.use_losses:
            results[key] = losses[key].item()

        self.iter_count += 1

        for key in self.epoch_keys:
            if key not in results:
                results.update({key: 0.0})

        self.training_step_outputs.append(results)
        return results

    def on_train_epoch_end(self):
        """Training epoch end."""
        outputs = self.training_step_outputs
        self.epoch_history["epoch"].append(self.current_epoch)
        self.epoch_history["mode"].append("train")

        for key in self.epoch_keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))
            self.log(
                key,
                self.epoch_history[key][-1],
                prog_bar=True,
                sync_dist=self.sync_dist,
            )

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs, losses = self(batch)
        results = {
            "predicts": outputs.detach().cpu(),
            "targets": batch["target"].detach().cpu(),
            "losses": {},
        }
        for key in self.use_losses:
            results["losses"][key] = losses[key].item()

        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        """Validation step end."""
        outputs = self.validation_step_outputs
        self.epoch_history["epoch"].append(self.current_epoch)
        self.epoch_history["mode"].append("valid")

        for key in self.use_losses:
            self.epoch_history[key].append(
                np.mean([output["losses"][key] for output in outputs])
            )
        self.metric.update(outputs)
        self.epoch_history["metric"].append(self.metric.compute())

        for key in self.epoch_keys:
            self.log(
                f"val_{key}",
                self.epoch_history[key][-1],
                prog_bar=True,
                sync_dist=self.sync_dist,
            )

        self.validation_step_outputs.clear()
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """Test step end."""
        self.epoch_history["epoch"].append(self.current_epoch)
        self.epoch_history["mode"].append("test")
        for key in self.epoch_keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))
            self.log(
                f"test_{key}",
                self.epoch_history[key][-1],
                prog_bar=True,
                sync_dist=self.sync_dist,
            )
