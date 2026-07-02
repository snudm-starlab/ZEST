"""
File Name   : main_two_stage_refine.py
Description : Main entry point for the ZEST hierarchical zero-shot quantization pipeline, managing progressive stages and data refinement.
Author      : Wonjin Cho (chowonjin0627@snu.ac.kr) Jeongin Yun (yji00828@snu.ac.kr) U Kang (ukang@snu.ac.kr), Seoul National University
"""


import logging
import torch
import torch.nn as nn
import os
import pickle
import fire

from models import get_model
from utils import get_dataset, evaluate_classifier
from reconstruct import quantize_model, reconstruct
from distill import distill_data
from distill_refine import refine_data_for_model
from reconstruction_for_quantization import reconstruct_w8a8_to_w4a4

# Configuration for logging
logging.basicConfig(
    style="{",
    format="{asctime} {levelname:8} {name:20} {message}",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

RECON_UNITS = (
    "BasicBlock",
    "Bottleneck",
    "ResBottleneckBlock",
    "DwsConvBlock",
    "ConvBlock",
    "InvertedResidual",
    "Linear",
    "Conv2d",
)


def run_quantization_stage(
    model_name,
    teacher_model,
    current_data,
    bit_w,
    bit_a,
    val_set,
    round_weight,
    recon_iter,
    recon_batch,
    is_first_stage=False,
):
    """Encapsulates a single quantization and reconstruction stage."""
    stage_name = f"W{bit_w}A{bit_a}"
    log.info(f"\n{'=' * 20} STAGE: {stage_name} {'=' * 20}")

    # 1. Create Model
    log.info(f"Creating {stage_name} quantized model...")
    fp32_base = get_model(model_name, pretrained=True).cuda().eval()
    student_model = quantize_model(
        fp32_base,
        bit_w=bit_w,
        bit_a=bit_a,
        quant_ops=(nn.Conv2d, nn.Linear, nn.Identity),
    )

    # 2. Reconstruct
    log.info(f"Reconstructing {stage_name} with teacher...")
    # Use the appropriate reconstruction function based on stage
    recon_fn = reconstruct if is_first_stage else reconstruct_w8a8_to_w4a4

    recon_fn(
        teacher_model,
        student_model,
        current_data,
        reconstruct_unit=RECON_UNITS,
        round_weight=round_weight,
        iterations=recon_iter,
        batch_size=recon_batch,
    )

    # 3. Evaluate
    accuracy = None
    if val_set:
        log.info(f"Evaluating {stage_name} model...")
        accuracy = evaluate_classifier(val_set, student_model)
        log.info(f"{stage_name} Model Accuracy: {accuracy * 100:.2f}%")

    # 4. Save Checkpoint
    checkpoint_path = f"{model_name}_{stage_name.lower()}_refined_quantized.pth"
    torch.save(
        {
            "model_state_dict": student_model.state_dict(),
            "model_name": model_name,
            "bit_w": bit_w,
            "bit_a": bit_a,
            "source": f"refined_distillation_{stage_name}",
        },
        checkpoint_path,
    )

    return student_model, accuracy, checkpoint_path


def main(
    val_path=r"D:\ImageNet\ILSVRC2012_img_train\val",
    model_name="resnet18",
    samples=1024,
    distill_batch=128,
    distill_iter=4000,
    lr_g=0.1,
    lr_z=0.01,
    refine_iter=500,
    lr_refine=0.01,
    recon_iter=20000,
    recon_batch=32,
    round_weight=1.0,
    target_bits=[(8, 8), (4, 4)],  # Easily change to [(8,8), (6,6), (4,4)] etc.
):
    os.makedirs("generated_data", exist_ok=True)
    val_set = get_dataset(val_path) if val_path and os.path.exists(val_path) else None

    # --- STAGE 0: FP32 Base ---
    log.info("STAGE 0: FP32 Setup")
    fp32_model = get_model(model_name, pretrained=True).cuda().eval()
    fp32_data_path = f"generated_data/{model_name}_fp32_synthetic.pickle"

    if os.path.exists(fp32_data_path):
        with open(fp32_data_path, "rb") as f:
            data_current = torch.from_numpy(pickle.load(f)[0]).cuda()
    else:
        data_current = distill_data(
            fp32_model, distill_batch, samples, lr_g, lr_z, distill_iter
        ).cuda()
        with open(fp32_data_path, "wb") as f:
            pickle.dump(
                [data_current.cpu().numpy()], f, protocol=pickle.HIGHEST_PROTOCOL
            )

    # --- PROGRESSIVE QUANTIZATION LOOP ---
    teacher_model = fp32_model
    results_summary = []

    for i, (bw, ba) in enumerate(target_bits):
        # 1. Run the Quantization Stage
        student, acc, path = run_quantization_stage(
            model_name,
            teacher_model,
            data_current,
            bw,
            ba,
            val_set,
            round_weight,
            recon_iter,
            recon_batch,
            is_first_stage=(i == 0),
        )
        results_summary.append((f"W{bw}A{ba}", acc, path))

        # 2. Refine data for the NEXT stage (if there is one)
        if i < len(target_bits) - 1:
            next_bw, next_ba = target_bits[i + 1]
            refine_path = f"generated_data/{model_name}_w{bw}a{ba}_refined.pickle"

            if os.path.exists(refine_path):
                log.info(f"Loading refined data: {refine_path}")
                with open(refine_path, "rb") as f:
                    data_current = torch.from_numpy(pickle.load(f)[0]).cuda()
            else:
                log.info(f"Refining data for next stage (W{next_bw}A{next_ba})...")
                data_current = refine_data_for_model(
                    model=student,
                    source_images=data_current,
                    lr=lr_refine,
                    iters=refine_iter,
                ).cuda()
                with open(refine_path, "wb") as f:
                    pickle.dump(
                        [data_current.cpu().numpy()],
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            # Current student becomes the teacher for the next round
            teacher_model = student

    # --- FINAL SUMMARY ---
    log.info("\n" + "=" * 80)
    log.info("QUANTIZATION COMPLETE")
    log.info("=" * 80)
    for name, acc, path in results_summary:
        acc_str = f"{acc * 100:.2f}%" if acc else "N/A"
        log.info(f"{name} Accuracy: {acc_str} | Saved: {path}")
    log.info("=" * 80)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    fire.Fire(main)
