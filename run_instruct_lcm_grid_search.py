"""
Grid Search Script for InstructDiffusion LCM-LoRA

Tests different combinations of guidance_scale and image_guidance_scale
to find optimal parameters for artifact-free editing.

Outputs:
- Images saved to folders named: instruct-diffusion-lcm_gs{guidance}_igs{image_guidance}
- Timing CSV file with all results
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import time
import csv
import sys
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler
from datasets import load_dataset
from utils.utils import txt_draw

# Add evaluation directory to path for metrics
sys.path.append('evaluation')
from matrics_calculator import MetricsCalculator


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


def load_instruct_diffusion_with_lcm(checkpoint_path, config_path):
    """
    Load InstructDiffusion checkpoint into diffusers pipeline and apply LCM-LoRA.
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
        original_config=config_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    print("Applying LCM-LoRA acceleration...")
    # Load LCM-LoRA adapter
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")

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
    image_guidance_scale=1.0
):
    """
    Edit image using InstructDiffusion with LCM acceleration.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for InstructDiffusion LCM parameters on HuggingFace dataset")
    parser.add_argument('--output_path', type=str, default="output/grid_search", help="Output directory")
    parser.add_argument('--checkpoint', type=str,
                       default="models/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt",
                       help="Path to InstructDiffusion checkpoint")
    parser.add_argument('--config', type=str,
                       default="models/InstructDiffusion/configs/instruct_diffusion.yaml",
                       help="Path to instruct_diffusion.yaml config")
    parser.add_argument('--steps', type=int, default=4,
                       help="Number of inference steps")
    parser.add_argument('--guidance_scales', nargs='+', type=float,
                       default=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                       help="Text guidance scales to test")
    parser.add_argument('--image_guidance_scales', nargs='+', type=float,
                       default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                       help="Image guidance scales to test")
    parser.add_argument('--num_images', type=int, default=700,
                       help="Number of random images to sample from dataset")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for dataset sampling")

    args = parser.parse_args()

    # Load HuggingFace dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset: timbrooks/instructpix2pix-clip-filtered")
    print(f"{'='*60}\n")
    dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train")

    # Sample random images
    random.seed(args.seed)
    np.random.seed(args.seed)

    total_images = len(dataset)
    num_to_sample = min(args.num_images, total_images)
    sampled_indices = random.sample(range(total_images), num_to_sample)

    print(f"Dataset size: {total_images}")
    print(f"Sampling {num_to_sample} random images (seed={args.seed})")
    print(f"{'='*60}\n")

    # Save sampled indices for reproducibility
    indices_save_path = os.path.join(args.output_path, "sampled_indices.json")
    os.makedirs(args.output_path, exist_ok=True)
    with open(indices_save_path, 'w') as f:
        json.dump({
            'seed': args.seed,
            'num_images': num_to_sample,
            'indices': sampled_indices
        }, f, indent=2)
    print(f"Saved sampled indices to: {indices_save_path}\n")

    # Load pipeline once (reuse for all parameter combinations)
    print(f"\n{'='*60}")
    print(f"Loading model...")
    print(f"{'='*60}\n")
    pipe = load_instruct_diffusion_with_lcm(args.checkpoint, args.config)

    # Prepare CSV for timing results
    timing_csv_path = os.path.join(args.output_path, "grid_search_timings.csv")
    os.makedirs(args.output_path, exist_ok=True)

    # Open CSV file and write header
    with open(timing_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'image_path',
            'guidance_scale',
            'image_guidance_scale',
            'steps',
            'elapsed_time_seconds'
        ])

        # Grid search over all parameter combinations
        total_combinations = len(args.guidance_scales) * len(args.image_guidance_scales)
        current_combination = 0

        for guidance_scale in args.guidance_scales:
            for image_guidance_scale in args.image_guidance_scales:
                current_combination += 1

                print(f"\n{'='*60}")
                print(f"Parameter Combination {current_combination}/{total_combinations}")
                print(f"  Guidance Scale: {guidance_scale}")
                print(f"  Image Guidance Scale: {image_guidance_scale}")
                print(f"  Steps: {args.steps}")
                print(f"{'='*60}\n")

                # Create folder name with parameters
                folder_name = f"instruct-diffusion-lcm_gs{guidance_scale}_igs{image_guidance_scale}"
                folder_path = os.path.join(args.output_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Process each sampled image
                for idx, dataset_idx in enumerate(sampled_indices):
                    item = dataset[dataset_idx]

                    # Extract image and instruction from HuggingFace dataset
                    input_image = item['input']  # PIL Image
                    editing_instruction_text = item['edit']  # str

                    # Create a unique filename for this image
                    image_filename = f"image_{dataset_idx:06d}.jpg"
                    present_image_save_path = os.path.join(folder_path, image_filename)

                    # Save input image temporarily for the editing function
                    temp_input_path = os.path.join(args.output_path, "temp_input.jpg")
                    input_image.save(temp_input_path)

                    print(f"Editing image {idx+1}/{num_to_sample} (dataset idx: {dataset_idx})")
                    print(f"  Instruction: {editing_instruction_text[:80]}...")
                    print(f"  -> {present_image_save_path}")

                    # Set seed for reproducibility
                    setup_seed()
                    torch.cuda.empty_cache()

                    # Track editing time
                    start_time = time.time()

                    # Run editing
                    edited_image = instruct_diffusion_edit_lcm(
                        pipe=pipe,
                        image_path=temp_input_path,
                        edit_prompt=editing_instruction_text,
                        num_inference_steps=args.steps,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale
                    )

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Save result
                    edited_image.save(present_image_save_path)

                    # Write timing to CSV
                    csv_writer.writerow([
                        f"dataset_idx_{dataset_idx}",
                        guidance_scale,
                        image_guidance_scale,
                        args.steps,
                        f"{elapsed_time:.2f}"
                    ])
                    csvfile.flush()  # Ensure data is written immediately

                    print(f"  Finished in {elapsed_time:.2f} seconds\n")

                # Clean up temp file
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)

    print(f"\n{'='*60}")
    print(f"Grid Search Complete!")
    print(f"Results saved to: {args.output_path}")
    print(f"Timing data saved to: {timing_csv_path}")
    print(f"{'='*60}\n")
