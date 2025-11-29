#!/usr/bin/env python3
"""
Summarize evaluation results from CSV file
"""
import csv
import numpy as np
import argparse

def summarize_results(csv_path, method_name="InstructDiffusion"):
    """
    Read evaluation CSV and compute summary statistics
    """
    ssim_values = []
    lpips_values = []
    clip_values = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get the column names (they contain the method name)
            ssim_col = [col for col in row.keys() if 'ssim' in col][0]
            lpips_col = [col for col in row.keys() if 'lpips' in col][0]
            clip_col = [col for col in row.keys() if 'clip_similarity_target_image' in col][0]

            # Parse values
            ssim_values.append(float(row[ssim_col]))
            lpips_values.append(float(row[lpips_col]))
            clip_values.append(float(row[clip_col]))

    # Calculate statistics
    total_images = len(ssim_values)
    ssim_mean = np.mean(ssim_values)
    ssim_std = np.std(ssim_values)
    lpips_mean = np.mean(lpips_values)
    lpips_std = np.std(lpips_values)
    clip_mean = np.mean(clip_values)
    clip_std = np.std(clip_values)

    # Print formatted results
    print(f"{method_name} Baseline Evaluation Results")
    print("=" * len(f"{method_name} Baseline Evaluation Results"))
    print()
    print(f"Total Images Processed: {total_images}")
    print()
    print("Metrics Summary:")
    print("-" * 16)
    print()
    print("SSIM (Structural Similarity):")
    print(f"Mean: {ssim_mean:.4f}")
    print(f"Std: {ssim_std:.4f}")
    print()
    print("LPIPS (Perceptual Distance):")
    print(f"Mean: {lpips_mean:.4f}")
    print(f"Std: {lpips_std:.4f}")
    print()
    print("CLIP Similarity (Semantic Alignment):")
    print(f"Mean: {clip_mean:.4f}")
    print(f"Std: {clip_std:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize evaluation results')
    parser.add_argument('--csv_path', type=str,
                        default='instructdiffusion_evaluation_results.csv',
                        help='Path to evaluation results CSV')
    parser.add_argument('--method_name', type=str,
                        default='InstructDiffusion',
                        help='Name of the method being evaluated')

    args = parser.parse_args()

    summarize_results(args.csv_path, args.method_name)
