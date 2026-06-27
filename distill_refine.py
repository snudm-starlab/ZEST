"""
File Name   : distill_refine.py
Description : Refines synthetic images via pixel-level optimization to match the quantized-domain constraints (BatchNorm statistics) of a target model.
Author      : Wonjin Cho (chowonjin0627@snu.ac.kr) Jeongin Yun (yji00828@snu.ac.kr) U Kang (ukang@snu.ac.kr), Seoul National University
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils import ActivationHook

log = logging.getLogger(__name__)

def l2_loss(A, B):
    return (A - B).norm()**2 / B.size(0)

def refine_data_for_model(model, source_images, lr=0.01, iters=500):
    """Refine existing synthetic images to match a target model's BN statistics.

    This approach directly optimizes the image pixels (no generator) to minimize
    the discrepancy between the image's activation statistics and the model's
    stored BatchNorm running_mean/running_var.

    Args:
        model: The target model (e.g., W8A8 quantized model) whose BN stats
               the images should match.
        source_images: A tensor of synthetic images (e.g., from FP32 distillation).
                       Shape: (N, 3, 224, 224).
        lr: Learning rate for pixel-level optimization.
        iters: Number of optimization iterations.

    Returns:
        Tensor: Refined synthetic images that match the target model's BN statistics.
    """
    eps = 1e-6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = copy.deepcopy(model).to(device).eval()
    except Exception:
        model = model.to(device).eval()
        for p in model.parameters():
            p.detach_()

    for param in model.parameters():
        param.requires_grad = False

    hooks, bn_stats = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(ActivationHook(module))
            bn_stats.append((
                module.running_mean.detach().clone().to(device),
                torch.sqrt(module.running_var + eps).detach().clone().to(device)
            ))

    refined_images = source_images.clone().to(device).requires_grad_(True)
    optimizer = optim.Adam([refined_images], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, patience=50)

    input_mean_target = torch.zeros(1, 3).to(device)
    input_std_target = torch.ones(1, 3).to(device)

    log.info(f"Starting image refinement for {refined_images.shape[0]} images...")

    for it in range(iters):
        optimizer.zero_grad()

        _ = model(refined_images)

        mean_loss, std_loss = 0.0, 0.0

        data_std, data_mean = torch.std_mean(refined_images, [2, 3])
        mean_loss += l2_loss(input_mean_target, data_mean)
        std_loss += l2_loss(input_std_target, data_std)

        for (bn_mean, bn_std), hook in zip(bn_stats, hooks):
            bn_input = hook.inputs
            data_std, data_mean = torch.std_mean(bn_input, [0, 2, 3])
            mean_loss += l2_loss(bn_mean, data_mean)
            std_loss += l2_loss(bn_std, data_std)

        total_loss = mean_loss + std_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        if (it + 1) % 50 == 0:
            log.info(f'Refine Iter {it+1}/{iters}, Loss: {total_loss:.4f}, Mean: {mean_loss:.4f}, Std: {std_loss:.4f}')

    for hook in hooks:
        hook.remove()

    return refined_images.detach()
