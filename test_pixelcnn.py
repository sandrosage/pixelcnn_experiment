import pytorch_lightning as pl
from pl_modules import PixelCNNModule, ReconstructKspaceDataModule
from pathlib import Path
from modules.kspace_data import KspaceDataTransform
from torchvision import transforms
from argparse import ArgumentParser

def build_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=Path,
        help="Path from which to load the model checkpoint",
    )

    parser.add_argument(
        "--root_dir",
        type=Path,
        help="Path to store the loggings",
    )

    args = parser.parse_args()
    return args

def cli_main(args):
    model = PixelCNNModule.load_from_checkpoint(args.ckpt_path)
    transform = transforms.Compose([
        transforms.Normalize(mean=(0.0,), std=(1.0,))  # Assume Laplace distributed inputs are mean 0, std 1
    ]) 
    dm = ReconstructKspaceDataModule(
            Path("D:/knee_dataset"),
            challenge="singlecoil",
            train_transform=KspaceDataTransform(),
            val_transform=KspaceDataTransform(),
            test_transform=KspaceDataTransform(),
            model_transform=transform,
            batch_size=1,
            num_workers=8,
            use_dataset_cache_file=True)
    trainer = pl.Trainer(default_root_dir=args.root_dir)
    trainer.test(model=model, datamodule=dm)

def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    print("-------------- TEST SCRIPT FOR MODEL EVALUATION --------------")
    run_cli()