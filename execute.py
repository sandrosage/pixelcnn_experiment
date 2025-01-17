from pl_modules import PixelCNNModule, ReconstructKspaceDataModule
import pytorch_lightning as pl
from pathlib import Path
from modules.kspace_data import KspaceDataTransform


if __name__ == "__main__":  
    model = PixelCNNModule()
    dm = ReconstructKspaceDataModule(
        Path("knee_dataset"),
        challenge="singlecoil",
        train_transform=KspaceDataTransform(),
        val_transform=KspaceDataTransform(),
        test_transform=KspaceDataTransform(),
        batch_size=1,
        num_workers=4)
    trainer = pl.Trainer(max_epochs=2, default_root_dir="./pixelcnn/")
    trainer.fit(model, dm)