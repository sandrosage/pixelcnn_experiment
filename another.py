from fastmri.pl_modules import FastMriDataModule, MriModule
from fastmri.data.transforms import UnetDataTransform, UnetSample, VarNetDataTransform, VarNetSample
from typing import NamedTuple, Optional, Dict
import torch
from fastmri.data.subsample import MaskFunc
import numpy as np
from fastmri.data import transforms as T
import fastmri
from fastmri.data.subsample import create_mask_for_mask_type
from pathlib import Path


class KspaceLDMSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        masked_kspace: Subsampled masked kspace 
        kspace: fully sampled (original) kspace
        target: The target image (if applicable).
    """

    masked_kspace: torch.Tensor
    kspace: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]

class KspaceLDMWithMaskInfoSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        masked_kspace: Subsampled masked kspace 
        kspace: fully sampled (original) kspace
        target: The target image (if applicable).
        acceleration: acceleration factor used by mask_func
        center_fraction: center fraction used by mask_func
    """

    masked_kspace: torch.Tensor
    kspace: torch.Tensor
    target: torch.Tensor
    acceleration: torch.Tensor
    center_fraction: torch.Tensor

class KspaceLDMDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """

        self.mask_func = mask_func
        self.use_seed = use_seed
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> KspaceLDMSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        masked_kspace = kspace_torch
        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

        return KspaceLDMSample(
            masked_kspace=masked_kspace,
            kspace =kspace_torch,
            target=target_torch

        )
            # sample = VarNetSample(
            #     masked_kspace=masked_kspace,
            #     mask=mask_torch.to(torch.bool),
            #     num_low_frequencies=num_low_frequencies,
            #     target=target_torch,
            #     fname=fname,
            #     slice_num=slice_num,
            #     max_value=max_value,
            #     crop_size=crop_size,
            # )
        # else:
        #     masked_kspace = kspace_torch
        #     shape = np.array(kspace_torch.shape)
        #     num_cols = shape[-2]
        #     shape[:-3] = 1
        #     mask_shape = [1] * len(shape)
        #     mask_shape[-2] = num_cols
        #     mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        #     mask_torch = mask_torch.reshape(*mask_shape)
        #     mask_torch[:, :, :acq_start] = 0
        #     mask_torch[:, :, acq_end:] = 0

        #     sample = VarNetSample(
        #         masked_kspace=masked_kspace,
        #         mask=mask_torch.to(torch.bool),
        #         num_low_frequencies=0,
        #         target=target_torch,
        #         fname=fname,
        #         slice_num=slice_num,
        #         max_value=max_value,
        #         crop_size=crop_size,
        #     )

        # return sample

if __name__ == "__main__":
    mask_type = "random"
    center_fractions = [0.08, 0.04]
    accelerations = [4, 8]
    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = KspaceLDMDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=Path("/vol/datasets/cil/2021_11_23_fastMRI_data/knee/unzipped"),
        challenge="multicoil",
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=True,
        test_split="test",
        sample_rate=None,
        batch_size=1,
        num_workers=4,
        distributed_sampler=False,
        use_dataset_cache_file=True
    )

    train_dl = data_module.train_dataloader()
    for batch in train_dl:
        print(batch._fields)
        print(batch.masked_kspace.shape)
        break