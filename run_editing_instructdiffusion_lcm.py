"""
LCM-Accelerated InstructDiffusion Editing Script

This script uses the diffusers library to load InstructDiffusion weights and applies
LCM-LoRA for 4-step fast inference (vs. 50 steps in the original).

Key differences from run_editing_instructdiffusion.py:
- Uses StableDiffusionInstructPix2PixPipeline instead of k-diffusion
- Applies LCM-LoRA for dramatic speedup (~12.5x faster)
- Maintains compatibility with PIE-Bench dataset
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import time
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler
from utils.utils import txt_draw


def mask_decode(encoded_mask, image_shape=[512, 512]):
    """Decode RLE-encoded mask to binary array."""
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # Avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def setup_seed(seed=1234):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_instruct_diffusion_with_lcm(checkpoint_path, config_path, use_lcm=True):
    """
    Load InstructDiffusion checkpoint into diffusers pipeline and apply LCM-LoRA.

    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Path to instruct_diffusion.yaml config
        use_lcm: Whether to apply LCM-LoRA acceleration

    Returns:
        Configured pipeline ready for inference
    """
    print(f"Loading InstructDiffusion checkpoint: {checkpoint_path}")
    print(f"Using config: {config_path}")

    # Convert to absolute path (required by from_single_file)
    checkpoint_path = os.path.abspath(checkpoint_path)
    config_path = os.path.abspath(config_path)

    # Check if files exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load checkpoint into diffusers format
    pipe = StableDiffusionInstructPix2PixPipeline.from_single_file(
        checkpoint_path,
        original_config=config_path,  # Use original_config instead of original_config_file
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    if use_lcm:
        print("Applying LCM-LoRA acceleration...")
        # Load LCM-LoRA adapter
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name = "lcm")  

        # Switch to LCM scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print("LCM-LoRA applied successfully!")

    return pipe


def instruct_diffusion_edit_lcm(
    pipe,
    image_path,
    edit_prompt,
    num_inference_steps=4,
    guidance_scale=1.5,
    image_guidance_scale=1.5
): 
    """
    Edit image using InstructDiffusion with LCM acceleration.

    Args:
        pipe: Configured diffusers pipeline
        image_path: Path to source image
        edit_prompt: Editing instruction
        num_inference_steps: Number of denoising steps (4 for LCM, 50 for standard)
        guidance_scale: Text CFG scale (use 1.0-2.0 for LCM, 5.0-7.5 for standard)
        image_guidance_scale: Image CFG scale

    Returns:
        PIL Image with source and edited image concatenated
    """
    # Load and prepare image
    input_image = Image.open(image_path).convert("RGB")

    # Resize to 512x512 (matching PIE-Bench dataset)
    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)

    # Run inference
    with torch.no_grad():
        edited_image = pipe(
            edit_prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale
        ).images[0]

    # Create output visualization
    image_instruct = txt_draw(f"edit prompt: {edit_prompt}")

    return Image.fromarray(
        np.concatenate(
            (
                image_instruct,
                np.array(input_image),
                np.zeros_like(image_instruct),
                np.array(edited_image)
            ),
            axis=1
        )
    )


image_save_paths = {
    "instruct-diffusion-lcm": "instruct-diffusion-lcm",
    "instruct-diffusion-standard": "instruct-diffusion-standard",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstructDiffusion editing with LCM acceleration")
    parser.add_argument('--rerun_exist_images', action="store_true", help="Rerun existing images")
    parser.add_argument('--data_path', type=str, default="data", help="Path to PIE-Bench dataset")
    parser.add_argument('--output_path', type=str, default="output", help="Output directory")
    parser.add_argument('--edit_category_list', nargs='+', type=str,
                       default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                       help="Editing categories to process")
    parser.add_argument('--edit_method_list', nargs='+', type=str,
                       default=["instruct-diffusion-lcm"],
                       help="Editing methods: instruct-diffusion-lcm or instruct-diffusion-standard")
    parser.add_argument('--checkpoint', type=str,
                       default="models/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt",
                       help="Path to InstructDiffusion checkpoint")
    parser.add_argument('--config', type=str,
                       default="models/InstructDiffusion/configs/instruct_diffusion.yaml",
                       help="Path to instruct_diffusion.yaml config")
    parser.add_argument('--steps', type=int, default=None,
                       help="Override number of inference steps (default: 4 for LCM, 50 for standard)")
    parser.add_argument('--guidance_scale', type=float, default=None,
                       help="Override text guidance scale (default: 1.5 for LCM, 5.0 for standard)")
    parser.add_argument('--image_guidance_scale', type=float, default=None,
                       help="Override image guidance scale (default: 1.0 for LCM, 1.5 for standard)")

    args = parser.parse_args()

    # Load dataset annotations
    with open(f"{args.data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    # Process each editing method
    for edit_method in args.edit_method_list:
        print(f"\n{'='*60}")
        print(f"Processing method: {edit_method}")
        print(f"{'='*60}\n")

        # Determine if using LCM
        use_lcm = "lcm" in edit_method.lower()

        # Set default parameters based on method
        if args.steps is not None:
            num_steps = args.steps
        else:
            num_steps = 4 if use_lcm else 50

        if args.guidance_scale is not None:
            guidance = args.guidance_scale
        else:
            guidance = 1.5 if use_lcm else 5.0

        if args.image_guidance_scale is not None:
            image_guidance = args.image_guidance_scale
        else:
            # Default: 1.0 for LCM (to reduce artifacts), 1.25 for standard (matching original script)
            image_guidance = 1.0 if use_lcm else 1.25

        print(f"Configuration:")
        print(f"  Use LCM: {use_lcm}")
        print(f"  Steps: {num_steps}")
        print(f"  Text Guidance Scale: {guidance}")
        print(f"  Image Guidance Scale: {image_guidance}")
        print()

        # Load pipeline (once per method)
        pipe = load_instruct_diffusion_with_lcm(
            args.checkpoint,
            args.config,
            use_lcm=use_lcm
        )

        # Process each image in dataset
        for key, item in editing_instruction.items():
            if item["editing_type_id"] not in args.edit_category_list:
                continue

            image_path = os.path.join(f"{args.data_path}/annotation_images", item["image_path"])
            editing_instruction_text = item["editing_instruction"]

            # Skip if source image doesn't exist
            if not os.path.exists(image_path):
                print(f"Warning: Source image not found, skipping: {image_path}")
                continue

            # Determine output path
            present_image_save_path = image_path.replace(
                args.data_path,
                os.path.join(args.output_path, image_save_paths[edit_method])
            )

            # Skip if already processed (unless rerun flag set)
            if os.path.exists(present_image_save_path) and not args.rerun_exist_images:
                print(f"Skip image [{image_path}] with [{edit_method}]")
                continue

            print(f"Editing image [{image_path}] with [{edit_method}]")

            # Set seed for reproducibility
            setup_seed()
            torch.cuda.empty_cache()

            # Track editing time
            start_time = time.time()

            # Run editing
            edited_image = instruct_diffusion_edit_lcm(
                pipe=pipe,
                image_path=image_path,
                edit_prompt=editing_instruction_text,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                image_guidance_scale=image_guidance
            )

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Save result
            os.makedirs(os.path.dirname(present_image_save_path), exist_ok=True)
            edited_image.save(present_image_save_path)

            print(f"Finished in {elapsed_time:.2f} seconds")

        print(f"\n{'='*60}")
        print(f"Completed method: {edit_method}")
        print(f"{'='*60}\n")
