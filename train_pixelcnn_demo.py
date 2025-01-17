from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
import os
from pl_modules import PixelCNNModule, ReconstructKspaceDataModule
from modules.kspace_data import KspaceDataTransform
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type

def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = Path("fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "pixelcnn" / "pixelcnn_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="equispaced_fraction",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = ReconstructKspaceDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        challenge="singlecoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = PixelCNNModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_channels=1, # of input channels
        n_layers=8,  # of gated conv layers
        hidden_channels=64,  #  of hidden conv channels
        lr=0.001,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    parser.add_argument(
        "--strategy",
        default=backend,
        type=str,
        help="what distributed version to use",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="what distributed version to use",
    )

    parser.add_argument(
        "--deterministic",
        default=True,
        type=bool,
        help="what distributed version to use",
    )

    parser.add_argument(
        "--default_root_dir",
        default=default_root_dir,
        type=Path,
        help="what distributed version to use",
    )

    parser.add_argument(
        "--max_epochs",
        default=50,
        type=int,
        help="what distributed version to use",
    )


    # parser = pl.Trainer.add_argparse_args(parser)
    # parser.set_defaults(
    #     gpus=num_gpus,  # number of gpus to use
    #     replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
    #     strategy=backend,  # what distributed version to use
    #     seed=42,  # random seed
    #     deterministic=True,  # makes things slower, but deterministic
    #     default_root_dir=default_root_dir,  # directory for logs and checkpoints
    #     max_epochs=50,  # max number of epochs
    # )

    args = parser.parse_args()
     # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if not hasattr(args, "resume_from_checkpoint"):
    # if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = KspaceDataTransform()
    val_transform = KspaceDataTransform()
    test_transform = KspaceDataTransform()
    # ptl data module - this handles data loaders
    data_module = ReconstructKspaceDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerations in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = PixelCNNModule(
        in_channels=args.in_channels,
        n_layers=args.n_layers,
        hidden_channels=args.hidden_channels,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer(
        deterministic=args.deterministic, 
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs
    )

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    print("Hello")
    run_cli()