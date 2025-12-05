"""
Run LCM-LoRA InstructDiffusion on a Single Image

Test LCM-LoRA with custom parameters on a single image.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, LCMScheduler


def main():
    parser = argparse.ArgumentParser(description="Run LCM InstructDiffusion on single image")
    parser.add_argument('--image_path', type=str, required=True,
                       help="Path to input image")
    parser.add_argument('--prompt', type=str, required=True,
                       help="Editing instruction")
    parser.add_argument('--output_path', type=str, default="output_lcm_test.jpg",
                       help="Output image path")
    parser.add_argument('--checkpoint', type=str,
                       default="models/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt",
                       help="Path to InstructDiffusion checkpoint")
    parser.add_argument('--config', type=str,
                       default="models/InstructDiffusion/configs/instruct_diffusion.yaml",
                       help="Path to config file")
    parser.add_argument('--steps', type=int, default=4,
                       help="Number of inference steps")
    parser.add_argument('--guidance_scale', type=float, default=5.0,
                       help="Text guidance scale")
    parser.add_argument('--image_guidance_scale', type=float, default=1.25,
                       help="Image guidance scale")
    parser.add_argument('--use_lcm', action='store_true', default=True,
                       help="Use LCM-LoRA (default: True)")

    args = parser.parse_args()

    print("="*60)
    print("LCM InstructDiffusion - Single Image")
    print("="*60)
    print(f"Image: {args.image_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Text Guidance: {args.guidance_scale}")
    print(f"Image Guidance: {args.image_guidance_scale}")
    print(f"Use LCM: {args.use_lcm}")
    print("="*60)

    # Load pipeline
    print("\nLoading model...")
    checkpoint_path = os.path.abspath(args.checkpoint)
    config_path = os.path.abspath(args.config)

    pipe = StableDiffusionInstructPix2PixPipeline.from_single_file(
        checkpoint_path,
        original_config=config_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    if args.use_lcm:
        print("Applying LCM-LoRA...")
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
        pipe.set_adapters(["lcm"], adapter_weights=[0.8])
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print("LCM-LoRA applied!")

    # Load and prepare image
    print(f"\nLoading image from {args.image_path}...")
    input_image = Image.open(args.image_path).convert("RGB")
    original_size = input_image.size
    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        edited_image = pipe(
            args.prompt,
            image=input_image,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale
        ).images[0]

    # Save result
    print(f"\nSaving to {args.output_path}...")
    edited_image.save(args.output_path)

    # Also save a side-by-side comparison
    comparison_path = args.output_path.replace('.jpg', '_comparison.jpg')
    input_resized = input_image
    comparison = Image.fromarray(
        np.concatenate([np.array(input_resized), np.array(edited_image)], axis=1)
    )
    comparison.save(comparison_path)

    print("="*60)
    print("Done!")
    print(f"Edited image: {args.output_path}")
    print(f"Comparison: {comparison_path}")
    print("="*60)


if __name__ == "__main__":
    main()
