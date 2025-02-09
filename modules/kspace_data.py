from typing import NamedTuple, Optional, Tuple, Dict
import torch
from fastmri.data.subsample import MaskFunc
import numpy as np
from fastmri.data.transforms import to_tensor, apply_mask
from fastmri import ifft2c, complex_abs
import fastmri.data.transforms as T

class TransformLabel(T.VarNetDataTransform):
    def __init__(self, mask_func = None, use_seed = True):
        super().__init__(mask_func, use_seed)
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.array,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> T.VarNetSample:
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
            # print("Target")
            # target_torch = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            # target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = T.VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=kspace_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = T.VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=(masked_kspace),
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )

        return sample

class KspaceSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """
    kspace : torch.Tensor
    masked_kspace: torch.Tensor
    reconstruction: torch.Tensor
    # num_low_frequencies: Optional[int]
    # target: torch.Tensor
    # fname: str
    # slice_num: int
    # max_value: float
    # crop_size: Tuple[int, int]


class KspaceDataTransform:
    """
    Data Transformer for training Kspace reconstruction models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
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
    ) -> KspaceSample:
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
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        slice_image = ifft2c(kspace_torch)           # Apply Inverse Fourier Transform to get the complex image
        reconstruction = complex_abs(slice_image)   # Compute absolute value to get a real image
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, _, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = KspaceSample(
                kspace=kspace_torch,
                masked_kspace=masked_kspace,
                reconstruction=reconstruction,
                # num_low_frequencies=num_low_frequencies,
                # target=target_torch,
                # fname=fname,
                # slice_num=slice_num,
                # max_value=max_value,
                # crop_size=crop_size,
            )
        else:
            sample = KspaceSample(
                kspace=kspace_torch,
                masked_kspace=kspace_torch,
                reconstruction=reconstruction
                # num_low_frequencies=0,
                # target=target_torch,
                # fname=fname,
                # slice_num=slice_num,
                # max_value=max_value,
                # crop_size=crop_size,
            )

        return sample