import configargparse

import configargparse
# In utils/configargparse_arguments.py
import configargparse

def build_configargparser(parser):
    model_group = parser.add_argument_group(title='Model options')
    dataset_group = parser.add_argument_group(title='Dataset options')
    module_group = parser.add_argument_group(title='Module options')
    trainer_group = parser.add_argument_group(title='Trainer options')

    # --- Base Trainer Arguments ---
    trainer_group.add_argument("--gpus", type=int, nargs='+', default=0)
    trainer_group.add_argument("--accelerator", type=str, default="ddp")
    trainer_group.add_argument("--resume_from_checkpoint", type=str, default=None)
    trainer_group.add_argument("--log_every_n_steps", type=int, default=50)
    trainer_group.add_argument("--log_interval", type=int, default=100)
    trainer_group.add_argument("--num_workers", type=int, default=8)
    trainer_group.add_argument("--num_sanity_val_steps", default=5, type=int)
    trainer_group.add_argument("--max_epochs", default=1000, type=int)
    trainer_group.add_argument("--min_epochs", default=1, type=int)
    trainer_group.add_argument("--check_val_every_n_epoch", default=1, type=int)
    trainer_group.add_argument("--save_top_k", default=1, type=int)
    trainer_group.add_argument("--early_stopping_metric", type=str, default="val_loss")
    trainer_group.add_argument("--log_save_interval", default=100, type=int)
    trainer_group.add_argument("--row_log_interval", default=100, type=int)
    trainer_group.add_argument("--fast_dev_run", default=False, type=str)
    trainer_group.add_argument("--name", default=None, type=str)
    trainer_group.add_argument("--on_polyaxon", action="store_true")
    trainer_group.add_argument("--output_path", type=str, default="logs")

    # --- Base Module / Model / Dataset Arguments ---
    module_group.add_argument("--module", type=str, required=True)
    model_group.add_argument("--model", type=str, required=True)
    dataset_group.add_argument("--data_root", default="", required=True, type=str)
    dataset_group.add_argument("--dataset", type=str, required=True)
    dataset_group.add_argument("--out_features", type=int, required=True)
    dataset_group.add_argument("--train_percent_check", type=float, default=1.0)
    dataset_group.add_argument("--val_percent_check", default=1.0, type=float)
    dataset_group.add_argument("--test_percent_check", default=1.0, type=float)
    dataset_group.add_argument("--overfit_pct", default=0.0, type=float)
    dataset_group.add_argument("--input_height", default=224, type=int)
    dataset_group.add_argument("--input_width", default=224, type=int)

    # --- Arguments from TeCNO module ---
    module_group.add_argument("--learning_rate", default=0.001, type=float)
    module_group.add_argument("--optimizer_name", default="adam", type=str)
    module_group.add_argument("--batch_size", default=1, type=int)
    module_group.add_argument("--weight_decay", default=1e-5, type=float, help="L2 regularization strength.")
    
    # --- Arguments from Cholec80 dataset ---
    dataset_group.add_argument("--features_per_seconds", default=25, type=float)
    dataset_group.add_argument("--features_subsampling", default=5, type=float)

    # --- Arguments from MSTCN model ---
    model_group.add_argument("--mstcn_stages", default=4, type=int)
    model_group.add_argument("--mstcn_layers", default=10, type=int)
    model_group.add_argument("--mstcn_f_maps", default=64, type=int)
    model_group.add_argument("--mstcn_f_dim", default=2048, type=int)
    model_group.add_argument("--mstcn_causal_conv", action='store_true')

    return parser