from pl_modules import PixelCNNModule, ReconstructKspaceDataModule
import pytorch_lightning as pl
from pathlib import Path
from modules.kspace_data import KspaceDataTransform
from modules.model import AdaptivePoolTransform, ZeroPaddingTransform
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.profilers import SimpleProfiler
import yaml

def main():
    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    
    cfg_dm = cfg["datamodule"]
    cfg_model = cfg["model"]
    cfg_trainer = cfg["trainer"]

    batch_size = cfg_dm["batch_size"]

    transform = transforms.Normalize(mean=(0.0,), std=(1.0,))  # Assume Laplace distributed inputs are mean 0, std 1

    if batch_size > 1:
        kspace_flag = cfg_dm["kspace"]
        assert (kspace_flag is not None), "the kspace_mode flag must be set if batchsize > 1"
        assert (kspace_flag["mode"] in ("adaptive", "padding")), "the kspace mode must either be 'adaptive' or 'padding'"
        if kspace_flag["mode"] == "adaptive":
            transform = transforms.Compose([
                AdaptivePoolTransform(output_size=(kspace_flag["size_x"], kspace_flag["size_y"]), pool_type=kspace_flag["pooling"]),
                transform
            ])
        else:
            transform = transforms.Compose([
                ZeroPaddingTransform(target_size=(kspace_flag["size_x"], kspace_flag["size_y"])),
                transform
            ])
            

    dm = ReconstructKspaceDataModule(
        data_path=Path(cfg_dm["data_path"]),
        challenge=cfg_dm["challenge"],
        batch_size=cfg_dm["batch_size"],
        num_workers=cfg_dm["num_workers"],
        use_dataset_cache_file=cfg_dm["use_dataset_cache_file"],
        train_transform=KspaceDataTransform(),
        val_transform=KspaceDataTransform(),
        test_transform=KspaceDataTransform(),
        model_transform=transform,
    )

    model_checkpoint = ModelCheckpoint(
        save_top_k=cfg_trainer["callbacks"]["model_ckpt"]["save_top_k"],
        monitor=cfg_trainer["callbacks"]["model_ckpt"]["monitor"],
        mode=cfg_trainer["callbacks"]["model_ckpt"]["mode"],
        filename="pixelcnn-kspace-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(
        max_epochs=cfg_trainer["max_epochs"],
        default_root_dir=cfg_trainer["default_root_dir"],
        profiler=SimpleProfiler(dirpath=cfg_trainer["default_root_dir"], filename="profiler_logs"),
        callbacks=[
            model_checkpoint, 
            DeviceStatsMonitor()
        ]
    )

    assert cfg["mode"] in ("train", "test"), "config.yaml mode must either be 'train' or 'test'"
    if cfg["mode"] == "train":
        print("-------------- TRAIN SCRIPT FOR MODEL TRAINING --------------")
        model = PixelCNNModule(
            in_channels=cfg_model["in_channels"],
            n_layers=cfg_model["n_layers"],
            hidden_channels=cfg_model["hidden_channels"],
            lr=cfg_model["lr"],
            test_criterion=cfg_model["test_criterion"],
            channel_mode=cfg_model["channel_mode"],
        )
        trainer.fit(model, dm)
    
    else:
        assert cfg_model["ckpt_path"] is not None, "for test mode the checkpoint path must be set"
        print("-------------- TEST SCRIPT FOR MODEL EVALUATION --------------".format(cfg_model["ckpt_path"]))
        print("-------------- LOAD MODEL FROM CHECKPOINT: {} --------------")
        model = PixelCNNModule.load_from_checkpoint(cfg_model["ckpt_path"])
        trainer.test(model,dm)
    

if __name__ == "__main__": 
    main()