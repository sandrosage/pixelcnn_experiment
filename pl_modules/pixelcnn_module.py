from modules.model import PixelCNN, LaplaceNLL
import pytorch_lightning as pl
from torch import optim, nn
from argparse import ArgumentParser
from typing import Literal, Optional
from torch.distributions import Laplace
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
            test_criterion: Optional[Literal["mse", "ssim"]] = None,
            channel_mode: Literal["real", "imag"] = "real",
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
        self.criterion = LaplaceNLL()
        self._channel_mode = channel_mode
        
        assert (test_criterion in ("mse", "ssim") or test_criterion is None), "test_criterion must be either 'mse', 'ssim' or None"

        if test_criterion == "mse":
            self.test_criterion = nn.MSELoss()
        elif test_criterion == "ssim":
            # ToDo
            self.test_criterion = None
        else:
            self.test_criterion = None
        
        print("TEST CRITERION: ", self.test_criterion if self.test_criterion is not None else "LaplaceNLL")
            
    
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
        input, target = rearrange_kspace(batch, self.mode)
        mean, log_scale = self(input)
        loss = self.criterion(mean=mean, log_scale=log_scale, target=target)
        self.log("train_loss", loss,  sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input, target = rearrange_kspace(batch, self.mode)
        mean, log_scale = self(input)
        if self.test_criterion is None:
            test_loss = self.criterion(mean=mean, log_scale=log_scale, target=target)
        else:
            sample = torch.zeros_like(target)
            _, _, h, w = sample.shape
            for i in range(h):
                for j in range(w):
                    mean, log_scale = self(sample)
                    scale = torch.exp(log_scale[:, :, i, j])  # Convert log_scale to scale
                    dist = Laplace(mean[:, :, i, j], scale)
                    sample[:, :, i, j] = dist.sample().clamp(0, 1)  # Sample and clamp values to [0, 1]
            test_loss = self.test_criterion(sample, target)
        
        self.log("test_loss", test_loss,  sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        input, target = rearrange_kspace(batch, self.mode)
        mean, log_scale = self(input)
        val_loss = self.criterion(mean=mean, log_scale=log_scale, target=target)
        self.log("val_loss", val_loss,  sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

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

        
