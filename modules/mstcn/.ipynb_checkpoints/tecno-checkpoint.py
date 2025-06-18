# import logging
# import torch
# from torch import optim
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import DataLoader
# from pytorch_lightning import LightningModule
# from utils.metric_helper import AccuracyStages
# from torchmetrics import Precision, Recall # Import directly from torchmetrics
# from torch import nn
# import numpy as np


# class TeCNO(LightningModule):
#     def __init__(self, hparams, model, dataset):
#         super().__init__()
#         # This is the correct way to save hyperparameters
#         self.save_hyperparameters(hparams, ignore=['model', 'dataset'])
        
#         self.dataset = dataset
#         self.model = model
#         self.batch_size = self.hparams.batch_size
        
#         # Loss function setup is correct
#         self.weights_train = np.asarray(self.dataset.weights["train"])
#         self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float())
        
#         # Initialize all stateful metrics
#         self.init_metrics()

#     def init_metrics(self):
#         # Central place to define all metrics for different splits (train/val/test)
#         num_classes = self.hparams.out_features
#         num_stages = self.hparams.mstcn_stages
        
#         # Metrics for the training set
#         self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
#         self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
#         self.train_acc_stages = AccuracyStages(num_stages=num_stages)

#         # Metrics for the validation set
#         self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
#         self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
#         self.val_acc_stages = AccuracyStages(num_stages=num_stages)
        
#         # Metrics for the test set
#         self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
#         self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
#         self.test_acc_stages = AccuracyStages(num_stages=num_stages)

#     def forward(self, x):
#         # Your forward pass is correct
#         video_fe = x.transpose(2, 1)
#         y_classes = self.model.forward(video_fe)
#         y_classes = torch.softmax(y_classes, dim=2)
#         return y_classes

#     def loss_function(self, y_classes, labels):
#         # Your loss function is correct
#         stages = y_classes.shape[0]
#         clc_loss = 0
#         for j in range(stages):
#             p_classes = y_classes[j].squeeze().transpose(1, 0)
#             ce_loss = self.ce_loss(p_classes, labels.squeeze())
#             clc_loss += ce_loss
#         clc_loss = clc_loss / (stages * 1.0)
#         return clc_loss
    
#     # The old helper methods are no longer needed and can be deleted:
#     # - get_class_acc
#     # - get_class_acc_each_layer
#     # - calc_precision_and_recall
#     # - log_average_precision_recall

#     # --- TRAINING LOOP ---
#     def training_step(self, batch, batch_idx):
#         features, true_phases, true_tools = batch
#         y_pred = self.forward(features)
        
#         # 1. Calculate and log loss
#         loss = self.loss_function(y_pred, true_phases)
#         self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

#         # 2. Update metrics (no calculation here, just updating the state)
#         final_stage_preds = y_pred[-1]
#         self.train_precision.update(final_stage_preds, true_phases)
#         self.train_recall.update(final_stage_preds, true_phases)
#         self.train_acc_stages.update(y_pred, true_phases)

#         # 3. Only return the loss
#         return loss

#     def on_train_epoch_end(self):
#         # On epoch end, compute the final metrics, log them, and reset for the next epoch
#         self.log("train_avg_precision", self.train_precision.compute(), sync_dist=True)
#         self.log("train_avg_recall", self.train_recall.compute(), sync_dist=True)

#         acc_stages = self.train_acc_stages.compute()
#         acc_stages_dict = {f"train_S{s+1}_acc": acc_stages[s] for s in range(len(acc_stages))}
#         acc_stages_dict["train_acc"] = acc_stages_dict.pop(f"train_S{len(acc_stages)}_acc")
#         self.log_dict(acc_stages_dict, sync_dist=True)
        
#         self.train_precision.reset()
#         self.train_recall.reset()
#         self.train_acc_stages.reset()

#     # --- VALIDATION LOOP ---
#     def validation_step(self, batch, batch_idx):
#         features, true_phases, true_tools = batch
#         y_pred = self.forward(features)
        
#         val_loss = self.loss_function(y_pred, true_phases)
#         self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

#         final_stage_preds = y_pred[-1]
#         self.val_precision.update(final_stage_preds, true_phases)
#         self.val_recall.update(final_stage_preds, true_phases)
#         self.val_acc_stages.update(y_pred, true_phases)
#         # We don't return anything from the validation_step

#     def on_validation_epoch_end(self):
#         # On epoch end, compute the final metrics, log them, and reset for the next epoch
#         self.log("val_avg_precision", self.val_precision.compute(), sync_dist=True)
#         self.log("val_avg_recall", self.val_recall.compute(), sync_dist=True)

#         acc_stages = self.val_acc_stages.compute()
#         metric_dict = {f"val_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
#         metric_dict["val_acc"] = metric_dict.pop(f"val_S{len(acc_stages)}_acc")
#         self.log_dict(metric_dict, sync_dist=True)
#         self.log("val: max acc last Stage", max(metric_dict.values())) # Simplified logging for best metric
        
#         self.val_precision.reset()
#         self.val_recall.reset()
#         self.val_acc_stages.reset()

#     # --- TEST LOOP ---
#     def test_step(self, batch, batch_idx):
#         features, true_phases, true_tools = batch
#         y_pred = self.forward(features)
        
#         test_loss = self.loss_function(y_pred, true_phases)
#         self.log("test_loss", test_loss, on_epoch=True)

#         final_stage_preds = y_pred[-1]
#         self.test_precision.update(final_stage_preds, true_phases)
#         self.test_recall.update(final_stage_preds, true_phases)
#         self.test_acc_stages.update(y_pred, true_phases)

#     def on_test_epoch_end(self):
#         self.log("test_avg_precision", self.test_precision.compute())
#         self.log("test_avg_recall", self.test_recall.compute())

#         acc_stages = self.test_acc_stages.compute()
#         metric_dict = {f"test_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
#         metric_dict["test_acc"] = metric_dict.pop(f"test_S{len(acc_stages)}_acc")
#         self.log_dict(metric_dict)
        
#         self.test_precision.reset()
#         self.test_recall.reset()
#         self.test_acc_stages.reset()

#     # --- OPTIMIZER ---
#     def configure_optimizers(self):
#         # Your optimizer setup is correct
#         optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#         return [optimizer]

#     # --- DATALOADERS ---
# #     # The dataloader methods are mostly fine, just removed a redundant `use_ddp` check
# #     def __dataloader(self, split=None):
# #         dataset = self.dataset.data[split]
# #         should_shuffle = (split == "train")
        
# #         sampler = None
# #         if self.trainer.strategy.is_global_zero:
# #              print(f"split: {split} - shuffle: {should_shuffle}")
        
# #         if self.trainer.is_global_zero and isinstance(self.trainer.strategy, "ddp"):
# #              sampler = DistributedSampler(dataset, shuffle=should_shuffle)
# #              should_shuffle = False # The sampler handles shuffling
        
# #         loader = DataLoader(
# #             dataset=dataset,
# #             batch_size=self.hparams.batch_size,
# #             shuffle=should_shuffle,
# #             sampler=sampler,
# #             num_workers=self.hparams.num_workers,
# #             pin_memory=True,
# #         )
# #         return loader
#     def __dataloader(self, split=None):
#         dataset = self.dataset.data[split]
#         should_shuffle = (split == "train")

#         sampler = None
#         # Correctly check which strategy is being used by inspecting the hparams
#         if self.hparams.accelerator == "ddp":
#             # The DistributedSampler is required for the 'ddp' strategy
#             # to ensure each GPU gets a unique slice of the data.
#             sampler = DistributedSampler(dataset, shuffle=should_shuffle)
#             # The sampler itself handles shuffling, so we turn off the loader's shuffle.
#             should_shuffle = False

#         # This print statement is helpful for debugging
#         if self.trainer.is_global_zero:
#             print(f"split: {split} - shuffle: {should_shuffle} - sampler: {'DDP' if sampler else 'None'}")

#         loader = DataLoader(
#             dataset=dataset,
#             batch_size=self.hparams.batch_size,
#             shuffle=should_shuffle,
#             sampler=sampler,
#             num_workers=self.hparams.num_workers,
#             pin_memory=True,
#         )
#         return loader

#     def train_dataloader(self):
#         return self.__dataloader(split="train")

#     def val_dataloader(self):
#         return self.__dataloader(split="val")

#     def test_dataloader(self):
#         return self.__dataloader(split="test")

#     # # This function remains unchanged
#     # @staticmethod
#     # def add_module_specific_args(parser):
#     #     # ...
#     #     return parser

import logging
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from utils.metric_helper import AccuracyStages
from torchmetrics import Precision, Recall
from torch import nn
import numpy as np


class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super().__init__()
        self.save_hyperparameters(hparams, ignore=['model', 'dataset'])
        
        self.dataset = dataset
        self.model = model
        self.batch_size = self.hparams.batch_size
        
        self.weights_train = np.asarray(self.dataset.weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float())
        
        self.init_metrics()

    def init_metrics(self):
        num_classes = self.hparams.out_features
        num_stages = self.hparams.mstcn_stages
        
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="micro")
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="micro")
        self.train_acc_stages = AccuracyStages(num_stages=num_stages, num_classes=num_classes)

        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="micro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="micro")
        self.val_acc_stages = AccuracyStages(num_stages=num_stages, num_classes=num_classes)
        
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="micro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="micro")
        self.test_acc_stages = AccuracyStages(num_stages=num_stages, num_classes=num_classes)

    def forward(self, x):
        video_fe = x.transpose(2, 1)
        # Return raw logits for the loss function
        return self.model.forward(video_fe)

    # def loss_function(self, y_classes, labels):
    #     stages = y_classes.shape[0]
    #     clc_loss = 0
    #     for j in range(stages):
    #         p_classes = y_classes[j].squeeze().transpose(1, 0)
    #         ce_loss = self.ce_loss(p_classes, labels.squeeze())
    #         clc_loss += ce_loss
    #     clc_loss = clc_loss / (stages * 1.0)
    #     return clc_loss
    
    def loss_function(self, y_classes, labels):
        # y_classes shape: (num_stages, batch_size, num_classes, seq_len)
        # labels shape: (batch_size, seq_len)
        stages = y_classes.shape[0]
        clc_loss = 0

        labels_flat = labels.flatten() # Shape: (batch_size * seq_len)

        for j in range(stages):
            p_classes_stage = y_classes[j] # Shape: (batch_size, num_classes, seq_len)
            # Reshape for CrossEntropyLoss, which expects (N, C)
            # Permute to (batch, seq, classes) -> Reshape to (batch * seq, classes)
            p_classes_reshaped = p_classes_stage.permute(0, 2, 1).reshape(-1, self.hparams.out_features)

            ce_loss = self.ce_loss(p_classes_reshaped, labels_flat)
            clc_loss += ce_loss

        clc_loss = clc_loss / stages
        return clc_loss
    
#     def _perform_step(self, batch, batch_idx, split: str):
#         features, true_phases, true_tools = batch
#         y_pred = self.forward(features)
        
#         # --- THIS IS THE FIX ---
#         # The model's output sequence is shorter than the input due to convolutions.
#         # We trim the ground truth tensors to match the prediction length.
#         pred_len = y_pred.shape[-1]
#         true_phases = true_phases[:, :pred_len]
#         true_tools = true_tools[:, :, :pred_len]
#         # -----------------------

#         loss = self.loss_function(y_pred, true_phases)
        
#         # Log loss
#         self.log(f"{split}_loss", loss, on_step=(split=="train"), on_epoch=True, prog_bar=True, sync_dist=True)

#         # Update metrics for the corresponding split
#         metrics_precision = getattr(self, f"{split}_precision")
#         metrics_recall = getattr(self, f"{split}_recall")
#         metrics_acc_stages = getattr(self, f"{split}_acc_stages")

#         final_stage_preds = torch.softmax(y_pred[-1], dim=1)
#         metrics_precision.update(final_stage_preds, true_phases)
#         metrics_recall.update(final_stage_preds, true_phases)
#         metrics_acc_stages.update(y_pred, true_phases)
        
#         return loss
    def _perform_step(self, batch, batch_idx, split: str):
        features, true_phases, true_tools = batch
        y_pred = self.forward(features) # This is now raw logits

        # Trim the ground truth tensors to match the prediction length
        pred_len = y_pred.shape[-1]
        true_phases = true_phases[:, :pred_len]
        true_tools = true_tools[:, :, :pred_len]

        # Calculate loss using the raw logits
        loss = self.loss_function(y_pred, true_phases)
        self.log(f"{split}_loss", loss, on_step=(split=="train"), on_epoch=True, prog_bar=True, sync_dist=True)

        # Get the logits from the final stage for other metrics
        final_stage_logits = y_pred[-1]

        # --- Update Metrics ---
        metrics_precision = getattr(self, f"{split}_precision")
        metrics_recall = getattr(self, f"{split}_recall")
        metrics_acc_stages = getattr(self, f"{split}_acc_stages")

        # AccuracyStages works directly on the multi-stage logits
        metrics_acc_stages.update(y_pred.detach(), true_phases)

        # Precision and Recall work on probabilities from the final stage
        final_stage_probs = torch.softmax(final_stage_logits, dim=1)
        metrics_precision.update(final_stage_probs, true_phases)
        metrics_recall.update(final_stage_probs, true_phases)

        if split == "train":
            return loss

    def _on_epoch_end(self, split: str):
        # Compute, log, and reset metrics for the corresponding split
        metrics_precision = getattr(self, f"{split}_precision")
        metrics_recall = getattr(self, f"{split}_recall")
        metrics_acc_stages = getattr(self, f"{split}_acc_stages")

        self.log(f"{split}_avg_precision", metrics_precision.compute(), sync_dist=True)
        self.log(f"{split}_avg_recall", metrics_recall.compute(), sync_dist=True)

        acc_stages = metrics_acc_stages.compute()
        acc_stages_dict = {f"{split}_S{s+1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        acc_stages_dict[f"{split}_acc"] = acc_stages_dict.pop(f"{split}_S{len(acc_stages)}_acc")
        self.log_dict(acc_stages_dict, sync_dist=True)
        
        metrics_precision.reset()
        metrics_recall.reset()
        metrics_acc_stages.reset()
    
    def training_step(self, batch, batch_idx):
        return self._perform_step(batch, batch_idx, "train")

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        self._perform_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._perform_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #     return [optimizer]
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay  # <-- Added this
        )
        return [optimizer]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = (split == "train")
        sampler = None
        if self.hparams.accelerator == "ddp":
            sampler = DistributedSampler(dataset, shuffle=should_shuffle)
            should_shuffle = False
        if self.trainer.is_global_zero:
            print(f"split: {split} - shuffle: {should_shuffle} - sampler: {'DDP' if sampler else 'None'}")
        loader = DataLoader(
            dataset=dataset, batch_size=self.hparams.batch_size, shuffle=should_shuffle,
            sampler=sampler, num_workers=self.hparams.num_workers, pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        return self.__dataloader(split="train")

    def val_dataloader(self):
        return self.__dataloader(split="val")

    def test_dataloader(self):
        return self.__dataloader(split="test")