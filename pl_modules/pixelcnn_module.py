from modules.model import PixelCNN, laplace_nll, rearrange_kspace, GatedPixelCNN, ResidualPixelCNN
import pytorch_lightning as pl
from torch import optim
from argparse import ArgumentParser
from pytorch_lightning.utilities.data import extract_batch_size
import torch


class PixelCNNModule(pl.LightningModule):
    def __init__(self, 
            in_channels: int = 1, 
            n_layers: int = 8, 
            hidden_channels: int = 64,
            lr: float = 3e-4,
            lr_step_size: int = 40,
            lr_gamma: float = 0.1,
            weight_decay: float = 0.0,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.model = PixelCNN(in_channels=self.in_channels, n_layers=self.n_layers, hidden_channels=self.hidden_channels)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, self.lr_step_size, self.lr_gamma
        )

        return [optimizer], [scheduler]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        target = rearrange_kspace(batch.kspace,0)
        input = rearrange_kspace(batch.masked_kspace,0)
        mean, log_scale = self(input)
        loss= laplace_nll(mean=mean, log_scale=log_scale, target=target)
        self.log("train_loss", loss,  sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("nll_loss", nll,  sync_dist=True, on_step=True, on_epoch=True)
        # self.log("l2_loss", l2_reg,  sync_dist=True, on_step=True, on_epoch=True)
        # Log GPU memory at any point
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        return loss

    def test_step(self, batch, batch_idx):
        target = rearrange_kspace(batch.kspace,0)
        input = rearrange_kspace(batch.masked_kspace,0)
        mean, log_scale = self(input)
        test_loss = laplace_nll(mean=mean, log_scale=log_scale, target=target)
        self.log("test_loss", test_loss,  sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        target = rearrange_kspace(batch.kspace,0)
        input = rearrange_kspace(batch.masked_kspace,0)
        mean, log_scale = self(input)
        val_loss = laplace_nll(mean=mean, log_scale=log_scale, target=target)
        self.log("val_loss", val_loss,  sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("val_nll_loss", nll,  sync_dist=True, on_step=True, on_epoch=True)
        # self.log("val_l2_loss", l2_reg,  sync_dist=True, on_step=True, on_epoch=True)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--in_channels",
            default=1,
            type=int,
            help="Number of input channels",
        )
        parser.add_argument(
            "--n_layers",
            default=8,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--hidden_channels",
            default=64,
            type=int,
            help="Number of hidden conv channels",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser

        
