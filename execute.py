from pl_modules import PixelCNNModule, ReconstructKspaceDataModule
import pytorch_lightning as pl
from pathlib import Path
from modules.kspace_data import KspaceDataTransform
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.profilers import SimpleProfiler
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    transform = transforms.Compose([
        transforms.Normalize(mean=(0.0,), std=(1.0,))  # Assume Laplace distributed inputs are mean 0, std 1
    ]) 

    dm = ReconstructKspaceDataModule(
        data_path=Path(cfg.datamodule.data_path),
        challenge=cfg.datamodule.challenge,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        use_dataset_cache_file=cfg.datamodule.use_dataset_cache_file,
        train_transform=KspaceDataTransform(),
        val_transform=KspaceDataTransform(),
        test_transform=KspaceDataTransform(),
        model_transform=transform,
    )

    model_checkpoint = ModelCheckpoint(
        save_top_k=cfg.trainer.callbacks.model_ckpt.save_top_k,
        monitor=cfg.trainer.callbacks.model_ckpt.monitor,
        mode=cfg.trainer.callbacks.model_ckpt.mode,
        filename="pixelcnn-kspace-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        profiler=SimpleProfiler(dirpath=cfg.trainer.default_root_dir, filename="profiler_logs"),
        callbacks=[
            model_checkpoint, 
            DeviceStatsMonitor()
        ]
    )

    assert cfg.mode in ("train", "test"), "config.yaml mode must either be 'train' or 'test'"
    if cfg.mode == "train":
        print("-------------- TRAIN SCRIPT FOR MODEL TRAINING --------------")
        model = PixelCNNModule(
            in_channels=cfg.model.in_channels,
            n_layers=cfg.model.n_layers,
            hidden_channels=cfg.model.hidden_channels,
            lr=cfg.model.lr,
            test_criterion=cfg.model.test_criterion,
            channel_mode=cfg.model.channel_mode,
        )
        trainer.fit(model, dm)
    
    elif cfg.mode == "test":
        assert cfg.model.ckpt_path is not None, "for test mode the checkpoint path must be set"
        print("-------------- TEST SCRIPT FOR MODEL EVALUATION --------------")
        print(f"-------------- LOAD MODEL FROM CHECKPOINT: {cfg.model.ckpt_path} --------------")
        model = PixelCNNModule.load_from_checkpoint(cfg.model.ckpt_path)
        trainer.test(model,dm)
    

if __name__ == "__main__": 
    main()