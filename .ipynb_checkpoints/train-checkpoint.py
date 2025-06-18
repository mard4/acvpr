# import configargparse
# from pathlib import Path
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
# import logging
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# from utils.utils import (
#     argparse_summary,
#     get_class_by_path,
# )
# from utils.configargparse_arguments import build_configargparser
# from datetime import datetime

# logging.disable(logging.WARNING)

# #SEED = 2334
# #torch.manual_seed(SEED)
# #np.random.seed(SEED)


# def train(hparams, ModuleClass, ModelClass, DatasetClass, logger):
#     """
#     Main training routine specific for this project
#     :param hparams:
#     """
#     # ------------------------
#     # 1 INIT LIGHTNING MODEL
#     # ------------------------
#     # This part is correct
#     model = ModelClass(hparams=hparams)
#     dataset = DatasetClass(hparams=hparams)
#     module = ModuleClass(hparams, model, dataset)

#     # ------------------------
#     # 3 INIT TRAINER
#     # ------------------------
    
#     # Using dirpath and filename is the modern way for ModelCheckpoint
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=f"{hparams.output_path}/checkpoints/",
#         filename=f"{hparams.name}-{{epoch:02d}}-{{{hparams.early_stopping_metric}:.2f}}",
#         save_top_k=hparams.save_top_k,
#         verbose=True,
#         monitor=hparams.early_stopping_metric,
#         mode='max'
#     )
    
#     early_stop_callback = EarlyStopping(
#         monitor=hparams.early_stopping_metric,
#         min_delta=0.00,
#         patience=3,
#         mode='max'
#     )

#     # Replaces the old `weights_summary` argument
#     summary_callback = ModelSummary(max_depth=1)

#     trainer = Trainer(
#         accelerator="gpu",
#         devices=hparams.gpus,
#         strategy=hparams.accelerator,
#         logger=logger,
#         fast_dev_run=hparams.fast_dev_run,
#         max_epochs=hparams.max_epochs,
#         callbacks=[early_stop_callback, checkpoint_callback, summary_callback],
#         num_sanity_val_steps=hparams.num_sanity_val_steps,
#         log_every_n_steps=hparams.log_every_n_steps
#     )
    
#     # ------------------------
#     # 4 START TRAINING
#     # ------------------------

#     # The checkpoint path for resuming is now passed to .fit()
#     trainer.fit(module, ckpt_path=hparams.resume_from_checkpoint)

#     print(
#         f"Best model score: {checkpoint_callback.best_model_score:.4f} at {checkpoint_callback.best_model_path}"
#     )
    
#     # ------------------------
#     # 5 START TESTING
#     # ------------------------
#     print("Testing with best model...")
#     # Using ckpt_path='best' automatically loads the best saved checkpoint
#     trainer.test(model=module, ckpt_path='best')
# if __name__ == "__main__":
#     # --- Stage 1: Initial Parse to discover module names ---
#     # Create a parser that will only look for the arguments needed to
#     # identify which modules to load. It ignores unknown arguments for now.
#     init_parser = configargparse.ArgParser(ignore_unknown_config_file_keys=True, add_help=False)
#     init_parser.add('-c', is_config_file=True, help='config file path')
#     init_parser.add_argument("--module", type=str, required=True)
#     init_parser.add_argument("--model", type=str, required=True)
#     init_parser.add_argument("--dataset", type=str, required=True)
#     hparams_temp, _ = init_parser.parse_known_args()

#     # --- Stage 2: Build the Full Parser ---
#     # Now that we know the module names, we can build a complete parser
#     # that knows about ALL possible arguments.

#     # Load the specific classes from the dynamically discovered names
#     ModuleClass = get_class_by_path(f"modules.{hparams_temp.module}")
#     ModelClass = get_class_by_path(f"models.{hparams_temp.model}")
#     DatasetClass = get_class_by_path(f"datasets.{hparams_temp.dataset}")

#     # Create the final parser
#     parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
#     parser.add('-c', is_config_file=True, help='config file path')

#     # Add all sets of arguments to it
#     parser = build_configargparser(parser)
#     parser = ModuleClass.add_module_specific_args(parser)
#     parser = ModelClass.add_model_specific_args(parser)
#     parser = DatasetClass.add_module_specific_args(parser)

#     # --- Stage 3: Final Parse ---
#     # Now parse all arguments from the command line and config file.
#     # This time, it will not encounter any "unrecognized" arguments.
#     hparams = parser.parse_args()

#     # --- This part remains the same as before ---
#     exp_name = (hparams.module.split(".")[-1] + "_" + hparams.dataset.split(".")[-1] + "_" + hparams.model.replace(".", "_"))
#     date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
#     hparams.name = date_str + exp_name
#     hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

#     tb_logger = TensorBoardLogger(str(hparams.output_path), name='tb')
#     wandb_logger = WandbLogger(name=hparams.name, project="tecno")

#     argparse_summary(hparams, parser)
#     print('Output path: ', hparams.output_path)

#     loggers = [tb_logger, wandb_logger]

#     train(hparams, ModuleClass, ModelClass, DatasetClass, loggers)

import configargparse
from pathlib import Path
from datetime import datetime
import logging

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary

from utils.utils import argparse_summary, get_class_by_path
from utils.configargparse_arguments import build_configargparser

logging.disable(logging.WARNING)

def train(hparams, ModuleClass, ModelClass, DatasetClass, logger):
    """
    Main training routine specific for this project
    """
    # 1. INIT LIGHTNING MODEL
    model = ModelClass(hparams=hparams)
    dataset = DatasetClass(hparams=hparams)
    module = ModuleClass(hparams, model, dataset)

    # 2. INIT CALLBACKS
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        filename=f"{hparams.name}-{{epoch:02d}}-{{{hparams.early_stopping_metric}:.2f}}",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='max'
    )
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=3,
        mode='max'
    )
    summary_callback = ModelSummary(max_depth=1)

    # 3. INIT TRAINER (with modern arguments)
    trainer = Trainer(
        accelerator="gpu",
        devices=hparams.gpus,
        strategy=hparams.accelerator,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback, summary_callback],
        max_epochs=hparams.max_epochs,
        fast_dev_run=hparams.fast_dev_run,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps
    )
    
    # 4. START TRAINING
    trainer.fit(module, ckpt_path=hparams.resume_from_checkpoint)

    print(f"Best model score: {checkpoint_callback.best_model_score:.4f} at {checkpoint_callback.best_model_path}")
    
    # 5. START TESTING
    print("Testing with best model...")
    trainer.test(model=module, ckpt_path='best')

    
if __name__ == "__main__":
    # Create the parser
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add('-c', is_config_file=True, help='config file path')

    # Add ALL arguments from our single helper function
    parser = build_configargparser(parser)

    # Now that the parser knows about every possible argument, parse once.
    hparams = parser.parse_args()

    # Load classes based on the now-known arguments
    ModuleClass = get_class_by_path(f"modules.{hparams.module}")
    ModelClass = get_class_by_path(f"models.{hparams.model}")
    DatasetClass = get_class_by_path(f"datasets.{hparams.dataset}")

    # --- Setup logging and paths ---
    exp_name = (hparams.module.split(".")[-1] + "_" + hparams.dataset.split(".")[-1] + "_" + hparams.model.replace(".", "_"))
    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
    hparams.name = date_str + exp_name
    hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    tb_logger = TensorBoardLogger(str(hparams.output_path), name='tb')
    wandb_logger = WandbLogger(name=hparams.name, project="tecno")

    argparse_summary(hparams, parser)
    print('Output path: ', hparams.output_path)

    loggers = [tb_logger, wandb_logger]

    # --- Run Training ---
    train(hparams, ModuleClass, ModelClass, DatasetClass, loggers)