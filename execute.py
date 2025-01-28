from pl_modules import PixelCNNModule, ReconstructKspaceDataModule
import pytorch_lightning as pl
from pathlib import Path
from modules.kspace_data import KspaceDataTransform
from torchvision import transforms



if __name__ == "__main__": 
    # Data Loaders
    transform = transforms.Compose([
        transforms.Normalize(mean=(0.0,), std=(1.0,))  # Assume Laplace distributed inputs are mean 0, std 1
    ]) 
    model = PixelCNNModule()
    dm = ReconstructKspaceDataModule(
        Path("../knee_dataset"),
        challenge="singlecoil",
        train_transform=KspaceDataTransform(),
        val_transform=KspaceDataTransform(),
        test_transform=KspaceDataTransform(),
        model_transform=transform,
        batch_size=1,
        num_workers=8,
        use_dataset_cache_file=True)
    trainer = pl.Trainer(max_epochs=25)
    trainer.fit(model, dm)