"""
Evaluate Grid Search Results

Calculates SSIM, LPIPS, and CLIP similarity metrics for each parameter combination.
Outputs one CSV per run with the same format as instructdiffusion_evaluation_results.csv,
plus timing information.
"""

import os
import json
import argparse
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Add evaluation directory to path
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


def extract_edited_image(concat_image):
    """
    Extract the edited image from concatenated output.
    Format: [instruction | source | spacer | edited]
    Each section is 512 pixels wide.
    """
    concat_array = np.array(concat_image)
    # The edited image is in the rightmost 512 pixels
    edited_image = Image.fromarray(concat_array[:, -512:, :])
    return edited_image


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    """Calculate a single metric using the same logic as evaluate.py"""
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "ssim_unedit_part":
        if (1-src_mask).sum() == 0 or (1-tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "lpips_unedit_part":
        if (1-src_mask).sum() == 0 or (1-tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)


def main():
    parser = argparse.ArgumentParser(description="Evaluate grid search results")
    parser.add_argument('--grid_search_path', type=str, default="output/grid_search",
                       help="Path to grid search output directory")
    parser.add_argument('--data_path', type=str, default="data",
                       help="Path to PIE-Bench dataset")
    parser.add_argument('--device', type=str, default="cuda",
                       help="Device for metric calculation")
    parser.add_argument('--edit_category_list', nargs='+', type=str,
                       default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                       help="Categories to evaluate")

    args = parser.parse_args()

    # Metrics to calculate
    METRICS = [
        "ssim",
        "ssim_unedit_part",
        "lpips",
        "lpips_unedit_part",
        "clip_similarity_source_image",
        "clip_similarity_target_image",
        "clip_similarity_target_image_edit_part"
    ]

    # Initialize metrics calculator
    print("Initializing metrics calculator...")
    metrics_calculator = MetricsCalculator(device=args.device)

    # Load dataset annotations
    print("Loading dataset annotations...")
    with open(f"{args.data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    # Load timing data if available
    timing_csv_path = os.path.join(args.grid_search_path, "grid_search_timings.csv")
    timing_data = {}
    if os.path.exists(timing_csv_path):
        print("Loading timing data...")
        with open(timing_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (float(row['guidance_scale']), float(row['image_guidance_scale']), row['image_path'])
                timing_data[key] = float(row['elapsed_time_seconds'])

    # Find all parameter combination folders
    print("Finding parameter combinations...")
    param_folders = []
    for folder_name in os.listdir(args.grid_search_path):
        folder_path = os.path.join(args.grid_search_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("instruct-diffusion-lcm_gs"):
            # Parse parameters from folder name
            # Format: instruct-diffusion-lcm_gs{guidance}_igs{image_guidance}
            parts = folder_name.replace("instruct-diffusion-lcm_", "").split("_")
            guidance_scale = float(parts[0].replace("gs", ""))
            image_guidance_scale = float(parts[1].replace("igs", ""))
            param_folders.append({
                'folder': folder_name,
                'guidance_scale': guidance_scale,
                'image_guidance_scale': image_guidance_scale,
                'path': folder_path
            })

    print(f"Found {len(param_folders)} parameter combinations\n")

    # Process each parameter combination
    for param_config in tqdm(param_folders, desc="Processing parameter combinations"):
        folder_name = param_config['folder']
        guidance_scale = param_config['guidance_scale']
        image_guidance_scale = param_config['image_guidance_scale']

        print(f"\nEvaluating: {folder_name}")

        # Output CSV for this run
        output_csv = os.path.join(args.grid_search_path, f"{folder_name}_evaluation_results.csv")

        with open(output_csv, 'w', newline='') as csvfile:
            # Header matching evaluate.py output format
            csv_writer = csv.writer(csvfile)
            header = ['image_path', 'editing_type'] + METRICS + ['elapsed_time_seconds']
            csv_writer.writerow(header)

            # Process each image
            for key, item in tqdm(editing_instruction.items(), desc=f"  Images", leave=False):
                if item["editing_type_id"] not in args.edit_category_list:
                    continue

                image_path = item["image_path"]
                category = item["editing_type_id"]

                # Path to source image
                source_image_path = os.path.join(args.data_path, "annotation_images", image_path)

                # Path to edited image (concatenated format)
                edited_image_path = os.path.join(param_config['path'], "annotation_images", image_path)

                # Skip if either image doesn't exist
                if not os.path.exists(source_image_path) or not os.path.exists(edited_image_path):
                    continue

                # Load images
                source_image = Image.open(source_image_path).convert("RGB")
                concat_image = Image.open(edited_image_path).convert("RGB")
                edited_image = extract_edited_image(concat_image)

                # Load mask and expand to 3 channels (same as original evaluate.py)
                mask_array = mask_decode(item["mask"])
                mask_array = mask_array[:, :, np.newaxis].repeat([3], axis=2)

                # Get prompts
                source_prompt = item["original_prompt"].replace("[", "").replace("]", "")
                target_prompt = item["editing_prompt"].replace("[", "").replace("]", "")

                # Calculate all metrics
                try:
                    row_data = [image_path, category]

                    for metric in METRICS:
                        value = calculate_metric(
                            metrics_calculator,
                            metric,
                            source_image,
                            edited_image,
                            mask_array,
                            mask_array,
                            source_prompt,
                            target_prompt
                        )
                        row_data.append(value)

                    # Add timing if available
                    timing_key = (guidance_scale, image_guidance_scale, source_image_path)
                    if timing_key in timing_data:
                        row_data.append(timing_data[timing_key])
                    else:
                        row_data.append("nan")

                    csv_writer.writerow(row_data)
                    csvfile.flush()

                except Exception as e:
                    print(f"Error processing {edited_image_path}: {e}")
                    continue

        print(f"  Saved to: {output_csv}")

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {args.grid_search_path}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
