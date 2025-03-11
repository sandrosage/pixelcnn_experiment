from typing import NamedTuple, Optional, Dict, Sequence, Union, Tuple
import torch
from fastmri.data.subsample import MaskFunc, RandomMaskFunc, temp_seed, EquispacedMaskFractionFunc, EquiSpacedMaskFunc, MagicMaskFractionFunc, MagicMaskFunc
import numpy as np
from fastmri.data.transforms import to_tensor

def take_kspace_channels(k: torch.Tensor, channel_mode: int):
    if channel_mode == -1:
        return k.permute(2,0,1)
    else:
        return k.permute(2,0,1)[0+channel_mode:1+channel_mode,:,:]

def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "random_info":
        return RandomMaskFuncWithInfo(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "magic":
        return MagicMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "magic_fraction":
        return MagicMaskFractionFunc(center_fractions, accelerations)
    else:
        raise ValueError(f"{mask_type_str} not supported")

def apply_mask_with_info(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies, mask_info = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies, mask_info

class RandomMaskFuncWithInfo(RandomMaskFunc):
    def __init__(self, center_fractions, accelerations, allow_any_combination = False, seed = None):
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
    
    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            # integrate mask_info: center_fraction + acceleration_factor
            center_mask, accel_mask, num_low_frequencies, mask_info = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies, mask_info
    
    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        mask_info = {"center_fraction": center_fraction, "acceleration": acceleration}

        return center_mask, acceleration_mask, num_low_frequencies, mask_info
    
class KspaceSample(NamedTuple):
    """
    A sample of masked k-space for pixelcnn model.

    Args:
        kspace: original unmasked k-space
        masked_kspace: k-space after applying sampling mask
        reconstruction: reconstructed real image
    """
    kspace : torch.Tensor
    masked_kspace: torch.Tensor
    reconstruction: torch.Tensor

class KspaceSampleMaskInfo(NamedTuple):
    """
    A sample of masked k-space for pixelcnn model.

    Args:
        kspace: original unmasked k-space
        masked_kspace: k-space after applying sampling mask
        reconstruction: reconstructed real image
        mask_info: center fraction + acceleration factor 
    """
    kspace : torch.Tensor
    masked_kspace: torch.Tensor
    reconstruction: torch.Tensor
    mask_info: dict


class KspaceDataTransform:
    """
    Data Transformer for training Kspace reconstruction models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True, channel_mode: int = -1):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        assert 1 >= channel_mode >= -1, "the channel_mode must either be '-1' for both channels (im + re), '0' for re and '1' for im channel"
        if channel_mode == -1:
            print(f"{self.__class__.__name__}: use both channels (im + re)")
        elif channel_mode == 1:
            print(f"{self.__class__.__name__}: use im channel")
        else:
            print(f"{self.__class__.__name__}: use re channel")

        self.channel_mode = channel_mode
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
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, _, _, mask_info = apply_mask_with_info(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = KspaceSampleMaskInfo(
                kspace=take_kspace_channels(kspace_torch, self.channel_mode),
                masked_kspace=take_kspace_channels(masked_kspace, self.channel_mode),
                reconstruction=target_torch,
                mask_info=mask_info
            )
        else:
            kspace_torch = take_kspace_channels(kspace_torch, self.channel_mode)
            sample = KspaceSample(
                kspace=kspace_torch,
                masked_kspace=kspace_torch,
                reconstruction=target_torch
            )
        return sample