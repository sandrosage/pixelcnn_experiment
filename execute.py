from pl_modules import PixelCNNModule, ReconstructKspaceDataModule
import pytorch_lightning as pl
from pathlib import Path
from modules.kspace_data import KspaceDataTransform, create_mask_for_mask_type
from modules.model import AdaptivePoolTransform, ZeroPaddingTransform, CenterCropTransform
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.profilers import SimpleProfiler
from omegaconf import OmegaConf

def main():
    cfg = OmegaConf.load("config.yaml")
    
    cfg_dm = cfg.datamodule
    cfg_model = cfg.model
    cfg_trainer = cfg.trainer

    batch_size = cfg_dm.batch_size

    transform = transforms.Normalize(mean=(0.0,), std=(1.0,))  # Assume Laplace distributed inputs are mean 0, std 1

    kspace_flag = cfg_dm.kspace
    if batch_size > 1:
        assert (kspace_flag is not None), "the kspace flag must be set if batchsize > 1"
    
    if kspace_flag is not None:
        assert (kspace_flag.mode in ("adaptive", "padding", "center_crop")), "the kspace mode must either be 'adaptive', 'padding' or 'center_crop'"
        if kspace_flag.mode == "adaptive":
            transform = transforms.Compose([
                AdaptivePoolTransform(output_size=(kspace_flag.size_x, kspace_flag.size_y), pool_type=kspace_flag.pooling),
                transform
            ])
        elif kspace_flag.mode == "padding":
            transform = transforms.Compose([
                ZeroPaddingTransform(target_size=(kspace_flag.size_x, kspace_flag.size_y)),
                transform
            ])
        else:
            transform = transforms.Compose([
                CenterCropTransform(target_size=(kspace_flag.size_x, kspace_flag.size_y)),
                transform
            ])
        

    print(f"MODEL TRANSFORM: {transform}")

    assert cfg_model.channel_mode in ("real", "imag", "both"), "channel_mode must either be 'real', 'imag' or 'both'"
    if cfg_model.channel_mode == "real":
        channel_mode = 0
    elif cfg_model.channel_mode == "imag":
        channel_mode = 1
    else:
        channel_mode = -1

    kspace_transform = KspaceDataTransform(channel_mode=channel_mode)
    if cfg_dm.mask_func is not None:
        mask_func = create_mask_for_mask_type(
            mask_type_str=cfg_dm.mask_func.type, 
            center_fractions=cfg_dm.mask_func.center_fractions, 
            accelerations=cfg_dm.mask_func.accelerations
            )
        kspace_transform = KspaceDataTransform(channel_mode=channel_mode, mask_func=mask_func)

    dm = ReconstructKspaceDataModule(
        data_path=Path(cfg_dm.data_path),
        challenge=cfg_dm.challenge,
        batch_size=cfg_dm.batch_size,
        num_workers=cfg_dm.num_workers,
        use_dataset_cache_file=cfg_dm.use_dataset_cache_file,
        train_transform=kspace_transform,
        val_transform=kspace_transform,
        test_transform=kspace_transform,
        model_transform=transform,
    )

    model_checkpoint = ModelCheckpoint(
        save_top_k=cfg_trainer.callbacks.model_ckpt.save_top_k,
        monitor=cfg_trainer.callbacks.model_ckpt.monitor,
        mode=cfg_trainer.callbacks.model_ckpt.mode,
        filename="pixelcnn-kspace-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(
        max_epochs=cfg_trainer.max_epochs,
        default_root_dir=cfg_trainer.default_root_dir,
        profiler=SimpleProfiler(dirpath=cfg_trainer.default_root_dir, filename="profiler_logs"),
        callbacks=[
            model_checkpoint, 
            DeviceStatsMonitor()
        ]
    )

    assert cfg.mode in ("train", "test"), "config.yaml mode must either be 'train' or 'test'"
    if cfg.mode == "train":
        print("-------------- TRAIN SCRIPT FOR MODEL TRAINING --------------")
        model = PixelCNNModule(
            in_channels=cfg_model.in_channels,
            n_layers=cfg_model.n_layers,
            hidden_channels=cfg_model.hidden_channels,
            lr=cfg_model.lr,
            test_criterion=cfg_model.test_criterion,
            channel_mode=cfg_model.channel_mode,
        )
        trainer.fit(model, dm)
    
    else:
        assert cfg_model["ckpt_path"] is not None, "for test mode the checkpoint path must be set"
        print("-------------- TEST SCRIPT FOR MODEL EVALUATION --------------")
        print("-------------- LOAD MODEL FROM CHECKPOINT: {} --------------".format(cfg_model.ckpt_path))
        model = PixelCNNModule.load_from_checkpoint(
            cfg_model.ckpt_path, 
            test_criterion=cfg_model.test_criterion, 
            channel_mode=cfg_model.channel_mode)
        trainer.test(model,dm)
    

if __name__ == "__main__": 
    main()