from pl_modules.data_module import MNISTDataModule
from pl_modules.pixelcnn_module import PixelCNNModule
from pytorch_lightning import Trainer
if __name__ == "__main__":
    model = PixelCNNModule()
    dm = MNISTDataModule("data")
    trainer = Trainer(max_epochs=5)
    trainer.fit(model=model, datamodule=dm)
