"""
File Name   : reconstruction_for_quantization.py
Description : Specialized block-wise reconstruction logic specifically tailored for transferring knowledge from an intermediate W8A8 TA to an ultra-low-bit W4A4 student.
Author      : Wonjin Cho (chowonjin0627@snu.ac.kr) Jeongin Yun (yji00828@snu.ac.kr) U Kang (ukang@snu.ac.kr), Seoul National University
"""



import logging
import torch
from torch import nn
from collections import OrderedDict
from utils import ActivationHook, find_parent
from quantizer import WeightQuantizer, ActivationQuantizer

log = logging.getLogger(__name__)

class LinearTempDecay:
    def __init__(self, iter_max, rel_start_decay, start_t, end_t):
        self.t_max = iter_max
        self.start_decay = rel_start_decay * iter_max
        self.start_b = start_t
        self.end_b = end_t

    def __call__(self, cur_iter):
        if cur_iter < self.start_decay:
            return self.start_b
        else:
            rel_t = (cur_iter-self.start_decay) / (self.t_max-self.start_decay)
            return self.end_b + (self.start_b-self.end_b)*max(0.0, 1 - rel_t)

def safe_pre_forward_hook(module, input):
    """Hook to safely set quantized weight before forward pass"""
    if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
        if hasattr(module, 'org_module'):
            # Store original weight
            if not hasattr(module, '_original_weight'):
                module._original_weight = module.org_module.weight.data.clone()
            # Set quantized weight with gradient tracking
            module.org_module.weight = module.weight_quantizer()

def safe_post_forward_hook(module, input, output):
    """Hook to restore original weight after forward pass"""
    if hasattr(module, '_original_weight'):
        if hasattr(module, 'org_module'):
            module.org_module.weight.data = module._original_weight
        delattr(module, '_original_weight')
    return output

def reconstruct_block_w8a8_to_w4a4(
    block_teacher, block_student, x_teacher, x_student, y_teacher, 
    iterations=20000, round_weight=1.0,
    lr_w_scale=0.0001, lr_a_scale=0.00004, lr_bit=0.001,
    annealing_range=(20,2), annealing_warmup=0.2, batch_size=32,
    tracker=None
):
    """Reconstruct W4A4 student block using W8A8 teacher block.
    
    Args:
        block_teacher: W8A8 quantized teacher block
        block_student: W4A4 quantized student block
        x_teacher: Input from W8A8 model
        x_student: Input from W4A4 model
        y_teacher: Output from W8A8 teacher block
        iterations: Number of optimization iterations
        round_weight: Weight for rounding loss
        lr_w_scale: Learning rate for weight scale
        lr_a_scale: Learning rate for activation scale
        lr_bit: Learning rate for bit allocation
        annealing_range: Temperature annealing range
        annealing_warmup: Warmup ratio for annealing
        batch_size: Batch size for reconstruction
    """
    # Set student block to training mode
    block_student.train()
    block_teacher.eval()
    
    # Register hooks to ensure quantized weights are used properly
    hooks = []
    for module in block_student.modules():
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            pre_hook = module.register_forward_pre_hook(safe_pre_forward_hook)
            post_hook = module.register_forward_hook(safe_post_forward_hook)
            hooks.extend([pre_hook, post_hook])
    
    # First, enable training mode for all quantizers and collect parameters
    param_a_scale = []
    param_w_scale = []
    param_bit = []

    # Collect parameters and ensure gradients are enabled
    # Also check for QuantizableLayer wrappers
    for name, module in block_student.named_modules():
        # Direct quantizer modules
        if isinstance(module, WeightQuantizer):
            module.train_mode = True
            if hasattr(module, 'scale') and module.scale is not None:
                module.scale.requires_grad_(True)
                param_w_scale.append(module.scale)
            if hasattr(module, 'bit_logit') and module.bit_logit is not None:
                module.bit_logit.requires_grad_(True)
                param_bit.append(module.bit_logit)
        elif isinstance(module, ActivationQuantizer):
            module.train_mode = True
            if hasattr(module, 'scale') and module.scale is not None:
                module.scale.requires_grad_(True)
                param_a_scale.append(module.scale)
        # Check for QuantizableLayer wrappers
        elif hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            wq = module.weight_quantizer
            wq.train_mode = True
            if hasattr(wq, 'scale') and wq.scale is not None:
                wq.scale.requires_grad_(True)
                param_w_scale.append(wq.scale)
            if hasattr(wq, 'bit_logit') and wq.bit_logit is not None:
                wq.bit_logit.requires_grad_(True)
                param_bit.append(wq.bit_logit)
        
        # Check for activation quantizers in QuantizableLayer
        if hasattr(module, 'act_quantizer') and module.act_quantizer is not None:
            aq = module.act_quantizer
            aq.train_mode = True
            if hasattr(aq, 'scale') and aq.scale is not None:
                aq.scale.requires_grad_(True)
                param_a_scale.append(aq.scale)

    log.info(f"Found {len(param_w_scale)} weight quantizers, {len(param_a_scale)} activation quantizers")
    
    # Build optimizer param groups
    param_groups = []
    if param_w_scale:
        param_groups.append({"params": param_w_scale, 'lr': lr_w_scale})
    if param_a_scale:
        param_groups.append({"params": param_a_scale, 'lr': lr_a_scale})
    
    if not param_groups:
        log.error(f"Block type: {type(block_student)}")
        log.error(f"Available modules: {[type(m).__name__ for m in block_student.modules()]}")
        raise ValueError("No quantizer parameters found in student block")
    
    opt_scale = torch.optim.Adam(param_groups)

    # Only create bit optimizer if there are bit parameters to optimize
    opt_bit = torch.optim.Adam(param_bit, lr=lr_bit) if param_bit else None
    scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingLR(opt_scale, T_max=iterations)

    temp_decay = LinearTempDecay(
        iterations, rel_start_decay=annealing_warmup,
        start_t=annealing_range[0], end_t=annealing_range[1])

    # Ensure reproducibility in reconstruction
    torch.manual_seed(42)

    iters = 0
    while iters < iterations:
        perms = torch.randperm(len(x_teacher)).view(batch_size, -1)
        for idx in perms:
            iters += 1

            # Use QDrop: mix quantized and input
            x_mix = torch.where(torch.rand_like(x_student[idx]) < 0.5, x_student[idx], x_teacher[idx])
            
            # Ensure input requires grad
            if not x_mix.requires_grad:
                x_mix.requires_grad_(True)
            
            # Forward pass with gradients enabled
            y_student = block_student(x_mix)
            
            # Check if output has grad_fn
            if y_student.grad_fn is None:
                raise RuntimeError(f"Student output has no grad_fn. Check if quantizers are being called in forward pass.")
            
            # Ensure y_teacher is detached (no gradients from teacher)
            y_target = y_teacher[idx].detach()
            
            # Reconstruction loss: match W8A8 teacher output
            recon_loss = (y_student - y_target).pow(2).sum(1).mean()
            round_loss = 0

            annealing_temp = temp_decay(iters)
            if iters >= annealing_warmup*iterations:
                for module in block_student.modules():
                    if isinstance(module, WeightQuantizer):
                        round_loss += (1 - (2*module.soft_target() - 1).abs().pow(annealing_temp)).sum()

            total_loss = recon_loss + round_loss * round_weight

            opt_scale.zero_grad()
            if opt_bit is not None:
                opt_bit.zero_grad()
            total_loss.backward()
            opt_scale.step()
            if opt_bit is not None:
                opt_bit.step()
            scheduler_scale.step()

            if iters == 1 or iters % 1000 == 0:
                log.info(
                    f'{iters}/{iterations}, Total loss: {total_loss:.3f} (rec:{recon_loss:.3f}, round:{round_loss:.3f})'
                    +f'\tb={annealing_temp:.2f}')
                
                # Support for evidence tracking
                if tracker is not None:
                    tracker.log(iters, total_loss)

            if iters >= iterations:
                break

    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Finish optimization, use hard rounding
    for module in block_student.modules():
        if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False


def reconstruct_w8a8_to_w4a4(teacher_w8a8, student_w4a4, cali_data, reconstruct_unit, **kwargs):
    """Reconstructs the W4A4 model using W8A8 model as teacher.

    Args:
        teacher_w8a8: W8A8 quantized teacher model
        student_w4a4: W4A4 quantized student model to reconstruct
        cali_data (tensor): calibration dataset
        reconstruct_unit (tuple): A list of block or layer to reconstruct
    """
    # Ensure models are in correct mode
    teacher_w8a8.eval()
    student_w4a4.train()
    
    teacher_modules = OrderedDict(teacher_w8a8.named_modules())
    student_modules = OrderedDict(student_w4a4.named_modules())
    reconstruct_pair = []

    visited = set()
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            # Find corresponding student module by name
            if name in student_modules:
                reconstruct_pair.append((module, student_modules[name], name))

    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):
        log.info(f'Reconstruct ({i+1}/{len(reconstruct_pair)}): {name}')
        
        # Check if student block has any quantizers
        has_quantizers = any(isinstance(m, (WeightQuantizer, ActivationQuantizer)) for m in student_block.modules())
        if not has_quantizers:
            log.warning(f'Skipping {name} - no quantizers found in student block')
            continue
        
        # Set blocks to correct modes
        teacher_block.eval()
        student_block.train()
        
        # Enable quantization for student
        for name, module in student_block.named_modules():
            if isinstance(module, ActivationQuantizer):
                module.train_mode = True

        # Disable quantization temporarily for activation collection
        for name, module in student_block.named_modules():
            if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = False
        
        act_x_teacher, act_y_teacher, act_x_student = [], [], []
        batch_size = 32
        cali_data_slices = cali_data.view(*(-1, batch_size, *cali_data.shape[1:]))

        t_hook = ActivationHook(teacher_block)
        s_hook = ActivationHook(student_block)

        with torch.no_grad():
            for x in cali_data_slices:
                teacher_w8a8(x)
                student_w4a4(x)
                act_x_teacher.append(t_hook.inputs)
                act_y_teacher.append(t_hook.outputs)
                act_x_student.append(s_hook.inputs)
        
        act_x_teacher = torch.cat(act_x_teacher)
        act_y_teacher = torch.cat(act_y_teacher)
        act_x_student = torch.cat(act_x_student)
                
        t_hook.remove()
        s_hook.remove()
        
        reconstruct_block_w8a8_to_w4a4(
            teacher_block, student_block, 
            act_x_teacher, act_x_student, act_y_teacher, 
            **kwargs
        )

    # Enable all quantization after reconstruction
    for name, module in student_w4a4.named_modules():
        if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False
