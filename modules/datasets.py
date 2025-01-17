from typing import Union
import torch
import fastmri.data.transforms as T
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from fastmri.data import SliceDataset
import os
from pathlib import Path

class filter_raw_sample():
                def __call__(self, raw_sample):
                    return True
    
class SingleCoilKnee(SliceDataset):
    def __init__(self, 
                root: Union[str, Path, os.PathLike], 
                noise_scheduler: Union[DDPMScheduler, DDIMScheduler] = None, 
                transform = None, 
                use_dataset_cache = False, 
                sample_rate = None, 
                volume_sample_rate = None, 
                dataset_cache_file = "dataset_cache.pkl", 
                num_cols = None, 
                raw_sample_filter = None):
        self.noise_scheduler = noise_scheduler
        challenge = "singlecoil"
        super().__init__(root, challenge, transform, use_dataset_cache, sample_rate, volume_sample_rate, dataset_cache_file, num_cols, raw_sample_filter)

    def __getitem__(self, i):
        item = super().__getitem__(i)   
        if self.noise_scheduler is not None:
            timestep = torch.IntTensor(torch.randint(low=0, high=self.noise_scheduler.timesteps.shape[0]-1, size=(1,)).item())
            item.masked_kspace = self.noise_scheduler.add_noise(original_samples = item.masked_kspace, noise= torch.randn_like(item.masked_kspace), timesteps=timestep)
        return item
    
class ReconstructKspaceDataset(SliceDataset):
    def __init__(self, 
                root, 
                challenge, 
                transform = None, 
                use_dataset_cache = False, 
                sample_rate = None, 
                volume_sample_rate = None, 
                dataset_cache_file = "dataset_cache.pkl", 
                num_cols = None, 
                raw_sample_filter = None):
        if raw_sample_filter is None:

            raw_sample_filter = filter_raw_sample()
        else:
            raw_sample_filter = raw_sample_filter
        
        super().__init__(root, challenge, transform, use_dataset_cache, sample_rate, volume_sample_rate, dataset_cache_file, num_cols, raw_sample_filter)