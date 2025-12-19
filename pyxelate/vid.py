from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import footprint_rectangle
from skimage.morphology import binary_dilation as skimage_dilation


class Vid:
    """Generator class that yields new images based on differences between them.

    This class is designed for processing video frames or image sequences,
    detecting keyframes and interpolating between them to reduce flickering
    in pixelated animations.

    Attributes:
        images: Sequence of images to process (each as numpy array).
        pad: Padding to crop from edges, as (top, bottom) or single int.
        sobel: Size multiplier for dilation footprint (actual size = sobel * 6).
        keyframe: Threshold for keyframe detection (0.0-1.0).
        sensitivity: Threshold for pixel change sensitivity (0.0-1.0).
    """

    def __init__(
        self,
        images: Sequence[NDArray[np.uint8]],
        pad: Union[int, Tuple[int, int], List[int], None] = 0,
        sobel: int = 2,
        keyframe: float = 0.3,
        sensitivity: float = 0.1,
    ) -> None:
        """Initialize Vid with a sequence of images.

        Args:
            images: List or tuple of image arrays (H, W, C) with values 0-255.
            pad: Rows to crop from top/bottom. Int applies to both, tuple for each.
            sobel: Dilation size multiplier for mask expansion.
            keyframe: Mean difference threshold to trigger new keyframe (0.0-1.0).
            sensitivity: Per-pixel difference threshold for mask (0.0-1.0).

        Raises:
            TypeError: If images is not a list or tuple.
            ValueError: If pad is not int or tuple of ints.
        """
        if not isinstance(images, (list, tuple)):
            raise TypeError(
                "Function only accepts list or tuple of image representations!"
            )
        self.images: Sequence[NDArray[np.uint8]] = images
        self.pad: Tuple[Optional[int], Optional[int]]
        if pad is None or pad == 0:
            self.pad = (None, None)
        elif isinstance(pad, int):
            self.pad = (pad, -pad)
        elif isinstance(pad, (list, tuple)):
            self.pad = (
                None if pad[0] == 0 else pad[0],
                None if pad[1] == 0 else -pad[1],
            )
        else:
            raise ValueError("The value of 'pad' must be int or (int, int)")
        self.sobel: int = int(sobel)
        self.keyframe: float = keyframe
        self.sensitivity: float = sensitivity

    def __iter__(self) -> Iterator[Tuple[NDArray[np.floating], bool]]:
        """Iterate over images, yielding processed frames with keyframe flags.

        Yields:
            Tuple of (processed_image, is_keyframe) where:
                - processed_image: Float array (H, W, 3) with values 0.0-1.0
                - is_keyframe: True if this frame is a keyframe, False otherwise

        Raises:
            ValueError: If an image has different dimensions than the first.
        """
        last_image: NDArray[np.floating]
        key_image: NDArray[np.floating]

        for i, image in enumerate(self.images):
            current_image: NDArray[np.floating] = np.clip(
                np.copy(image[self.pad[0] : self.pad[1], :, :3]) / 255.0, 0.0, 1.0
            )
            if i == 0:
                last_image = np.copy(current_image)
                key_image = np.copy(current_image)
                yield last_image, True
            else:
                if not np.all(
                    [a == b for a, b in zip(last_image.shape, current_image.shape)]
                ):
                    raise ValueError(f"Image at position {i} has different size!")
                last_difference: NDArray[np.floating] = np.abs(
                    current_image[:, :, :3] - last_image[:, :, :3]
                )
                last_difference = np.max(last_difference, axis=2)
                key_difference: NDArray[np.floating] = np.abs(
                    current_image[:, :, :3] - key_image[:, :, :3]
                )
                key_difference = np.max(key_difference, axis=2)
                if (
                    np.mean(last_difference) < self.keyframe
                    or np.mean(key_difference) < self.keyframe
                ):
                    mask: NDArray[np.bool_] = np.where(
                        key_difference > self.sensitivity, True, False
                    )
                    if self.sobel:
                        mask = skimage_dilation(
                            mask,
                            footprint=footprint_rectangle(
                                (self.sobel * 6, self.sobel * 6)
                            ),
                        )
                    mask_expanded: NDArray[np.bool_] = np.expand_dims(mask, axis=-1)
                    last_image = current_image * mask_expanded + last_image * (
                        1.0 - mask_expanded
                    )
                    yield last_image, False
                else:
                    # new keyframe
                    last_image = np.copy(current_image)
                    key_image = np.copy(current_image)
                    yield last_image, True
