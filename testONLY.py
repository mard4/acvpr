import configargparse
from pathlib import Path
from pytorch_lightning import Trainer
from utils.utils import get_class_by_path
from utils.configargparse_arguments import build_configargparser
import torch

def test(hparams, ModuleClass, ModelClass, DatasetClass, ckpt_path):
    """
    Main testing routine to load a checkpoint and run the test loop.
    """
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    # Load the LightningModule from the checkpoint file.
    module = ModuleClass.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams=hparams,
        model=ModelClass(hparams=hparams),
        dataset=DatasetClass(hparams=hparams)
    )

    # Initialize a Trainer instance
    #trainer = Trainer(gpus=hparams.gpus)
    #trainer = Trainer(gpus=hparams.gpus, accelerator=hparams.accelerator)
    trainer = Trainer(accelerator="gpu", devices=hparams.gpus, strategy=hparams.accelerator) #new version 
    # Run the test
    trainer.test(model=module)
    print("Testing finished.")

if __name__ == "__main__":
    # --- Parse all arguments from config and command line ---
    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    
    # Add a specific command-line argument to point to the saved model
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file (.ckpt) to test.')
    
    # Use the same function as train.py to get all base arguments
    parser, hparams = build_configargparser(parser)

    # Allow the specific module, model, and dataset to add their own arguments
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)
    
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)
    
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)

    # Parse all arguments again to include the ones from the modules
    hparams = parser.parse_args()

    # --- Correctly set the output_path as a Path object ---
    ckpt_path = Path(hparams.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    
    # The output directory is two levels above the checkpoint file 
    # (e.g., .../logs/EXP_NAME/checkpoints/model.ckpt)
    hparams.output_path = ckpt_path.parent.parent
    
    # --- Run the testing function ---
    test(hparams, ModuleClass, ModelClass, DatasetClass, str(ckpt_path))