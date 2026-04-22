import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor_to_numpy_img(tensor):
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    return img


def compute_psnr_ssim(pred, target):
    pred_np = tensor_to_numpy_img(pred)
    target_np = tensor_to_numpy_img(target)
    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=255)
    ssim = structural_similarity(target_np, pred_np, channel_axis=2, data_range=255)
    return psnr, ssim
