import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

from basicsr.utils.registry import METRIC_REGISTRY

from ssr.utils.metric_utils import to_y_channel, reorder_image

def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

@METRIC_REGISTRY.register()
def calculate_cssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, multi_ch=True, **kwargs):

    img1 = img
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Try different offsets of the images.
    # We will crop img1 so top-left is at (row_offset, col_offset),
    # and img2 so top-left is at (max_offset - row_offset, max_offset - col_offset).
    max_offset = 8
    height, width = img1.shape[0], img1.shape[1]
    crop_height, crop_width = height - max_offset, width - max_offset
    best_ssim = -1
    for row_offset in range(max_offset+1):
        for col_offset in range(max_offset+1):
            cur_img1 = img1[row_offset:, col_offset:]
            cur_img1 = cur_img1[0:crop_height, 0:crop_width].copy()
            cur_img2 = img2[(max_offset-row_offset):, (max_offset-col_offset):]
            cur_img2 = cur_img2[0:crop_height, 0:crop_width].copy()

            # Compute bias to minimize as the average pixel value difference of each channel.
            for channel_idx in range(img1.shape[2]):
                bias = np.mean(cur_img1[:, :, channel_idx] - cur_img2[:, :, channel_idx])
                cur_img2[:, :, channel_idx] += bias

            # Now compute SSIM.
            ssim = _ssim(cur_img1, cur_img2)
            if ssim > best_ssim:
                best_ssim = ssim

    return best_ssim