import pytorch_lightning as pl
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from fastmri.data.subsample import MaskFunc
from typing import Optional, Callable, Union, Literal
import os
from modules.datasets import SingleCoilKnee, ReconstructKspaceDataset
from fastmri.pl_modules import FastMriDataModule
from fastmri.data import SliceDataset, CombinedSliceDataset, VolumeSampler
from pathlib import Path

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset, ReconstructKspaceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str ="./", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.0,), std=(1.0,))])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            train_set_size = int(len(mnist_full) * 0.8)
            valid_set_size = len(mnist_full) - train_set_size
            self.mnist_train, self.mnist_val = random_split(mnist_full, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(42))
        
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
    


class KneeSingleCoilDataModule(pl.LightningDataModule):
    def __init__(self, kspace_transform, data_dir: str ="./", batch_size: int = 1, mask_func: Optional[MaskFunc] = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mask_func = mask_func
        self.kspace_transform = kspace_transform(self.mask_func)
    
    def prepare_data(self):
        for dir in os.listdir(self.data_dir):
            if "_train" in dir:
                self.train_data_dir = os.path.join(self.data_dir, dir)

            elif "_val" in dir:
                self.val_data_dir = os.path.join(self.data_dir, dir)

            elif "_test" in dir:
                self.test_data_dir = os.path.join(self.data_dir, dir)
    
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train = SingleCoilKnee(root=self.train_data_dir, transform=self.kspace_transform)
            self.dataset_val = SingleCoilKnee(root=self.val_data_dir, transform=self.kspace_transform)
        
        if stage == "test":
            self.dataset_test = SingleCoilKnee(root=self.test_data_dir, transform=self.kspace_transform)

        if stage == "test":
            self.dataset_test = SingleCoilKnee(root=self.test_data_dir, transform=self.kspace_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
    

class ReconstructKspaceDataModule(FastMriDataModule):
    def __init__(self, 
        data_path: Path, 
        challenge: str, 
        train_transform: Callable, 
        val_transform: Callable, 
        test_transform: Callable, 
        model_transform: Callable,
        combine_train_val: bool = False, 
        test_split: str = "test", 
        test_path: Optional[Path] = None, 
        sample_rate: Optional[float]= None, 
        val_sample_rate: Optional[float]= None, 
        test_sample_rate: Optional[float]= None, 
        volume_sample_rate: Optional[float]= None, 
        val_volume_sample_rate: Optional[float]= None, 
        test_volume_sample_rate: Optional[float]= None, 
        train_filter: Optional[Callable] = None, 
        val_filter:  Optional[Callable] = None, 
        test_filter: Optional[Callable] = None, 
        use_dataset_cache_file: bool = True, 
        batch_size: int = 1, 
        num_workers: int = 4, 
        distributed_sampler: bool = False
    ):
        
        super().__init__(data_path, 
            challenge, 
            train_transform, 
            val_transform, 
            test_transform, 
            combine_train_val, 
            test_split, 
            test_path, 
            sample_rate, 
            val_sample_rate, 
            test_sample_rate, 
            volume_sample_rate, 
            val_volume_sample_rate, 
            test_volume_sample_rate, 
            train_filter, 
            val_filter, 
            test_filter, 
            use_dataset_cache_file, 
            batch_size, 
            num_workers, 
            distributed_sampler)
        
        self.model_transform = model_transform
    
    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                sample_rate = (
                    self.test_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.test_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.test_filter

        # if desired, combine train and val together for the train split
        dataset: Union[SliceDataset, CombinedSliceDataset, ReconstructKspaceDataset]
        # TODO: Find out what CombinedSliceDataset is and how to model it for the ReconstructKspaceDataset
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{self.challenge}_{data_partition}"

            dataset = ReconstructKspaceDataset(
                root=data_path,
                transform=data_transform,
                model_transform=self.model_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )
            # dataset = SliceDataset(
            #     root=data_path,
            #     transform=data_transform,
            #     sample_rate=sample_rate,
            #     volume_sample_rate=volume_sample_rate,
            #     challenge=self.challenge,
            #     use_dataset_cache=self.use_dataset_cache_file,
            #     raw_sample_filter=raw_sample_filter,
            # )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
            persistent_workers=True
        )

        return dataloader
    
    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / f"{self.challenge}_test"
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                _ = ReconstructKspaceDataset(
                    root=data_path,
                    transform=data_transform,
                    model_transform=self.model_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                )
                # _ = SliceDataset(
                #     root=data_path,
                #     transform=data_transform,
                #     sample_rate=sample_rate,
                #     volume_sample_rate=volume_sample_rate,
                #     challenge=self.challenge,
                #     use_dataset_cache=self.use_dataset_cache_file,
                # )