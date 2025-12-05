"""
Summarize Grid Search Results

Creates summary text files (one per run) and a combined CSV with average metrics.
"""

import os
import csv
import argparse
import pandas as pd


def create_summary_text(csv_path, output_txt_path, guidance_scale, image_guidance_scale):
    """Create a text summary file for a single run (matching instructdiffusion format)."""
    df = pd.read_csv(csv_path)

    # Calculate mean for each metric (excluding nan values)
    metrics = [col for col in df.columns if col not in ['image_path', 'editing_type']]

    # Category names mapping
    category_names = {
        '0': 'Random editing',
        '1': 'Change object',
        '2': 'Add object',
        '3': 'Delete object',
        '4': 'Change attribute (content)',
        '5': 'Change attribute (pose)',
        '6': 'Change attribute (color)',
        '7': 'Change attribute (material)',
        '8': 'Change background',
        '9': 'Change style'
    }

    with open(output_txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Grid Search Results Summary\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Image Guidance Scale: {image_guidance_scale}\n")
        f.write("="*60 + "\n\n")

        # Overall average
        f.write("Average Metrics Across All Images:\n")
        f.write("-"*60 + "\n")

        for metric in metrics:
            # Convert to numeric, coercing errors to NaN
            values = pd.to_numeric(df[metric], errors='coerce')
            mean_value = values.mean()
            std_value = values.std()

            if pd.isna(mean_value):
                f.write(f"{metric:40s}: nan\n")
            else:
                f.write(f"{metric:40s}: {mean_value:.6f} ± {std_value:.6f}\n")

        f.write("\n")
        f.write("="*60 + "\n")
        f.write(f"Total images evaluated: {len(df)}\n")
        f.write("="*60 + "\n\n")

        # Breakdown by category
        f.write("="*60 + "\n")
        f.write("Breakdown by Editing Category\n")
        f.write("="*60 + "\n\n")

        for category_id in sorted(df['editing_type'].unique()):
            category_df = df[df['editing_type'] == category_id]
            category_name = category_names.get(str(category_id), f"Category {category_id}")

            f.write(f"Category {category_id}: {category_name}\n")
            f.write(f"  Images: {len(category_df)}\n")
            f.write("-"*60 + "\n")

            for metric in metrics:
                values = pd.to_numeric(category_df[metric], errors='coerce')
                mean_value = values.mean()
                std_value = values.std()

                if pd.isna(mean_value):
                    f.write(f"  {metric:38s}: nan\n")
                else:
                    f.write(f"  {metric:38s}: {mean_value:.6f} ± {std_value:.6f}\n")

            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize grid search evaluation results")
    parser.add_argument('--grid_search_path', type=str, default="output/grid_search",
                       help="Path to grid search output directory")
    parser.add_argument('--output_csv', type=str, default="output/grid_search/summary.csv",
                       help="Output CSV with all summaries")

    args = parser.parse_args()

    print("="*60)
    print("Grid Search Summary Generator")
    print("="*60)

    # Find all evaluation result CSVs
    eval_csvs = []
    for filename in os.listdir(args.grid_search_path):
        if filename.endswith("_evaluation_results.csv"):
            # Handle standard InstructDiffusion separately
            if filename == "instruct-diffusion-standard_evaluation_results.csv":
                eval_csvs.append({
                    'filename': filename,
                    'path': os.path.join(args.grid_search_path, filename),
                    'guidance_scale': 5.0,  # Standard defaults
                    'image_guidance_scale': 1.25,
                    'method': 'standard'
                })
            else:
                # Parse parameters from filename
                # Format: instruct-diffusion-lcm_gs{guidance}_igs{image_guidance}_evaluation_results.csv
                parts = filename.replace("instruct-diffusion-lcm_", "").replace("_evaluation_results.csv", "").split("_")
                guidance_scale = float(parts[0].replace("gs", ""))
                image_guidance_scale = float(parts[1].replace("igs", ""))

                eval_csvs.append({
                    'filename': filename,
                    'path': os.path.join(args.grid_search_path, filename),
                    'guidance_scale': guidance_scale,
                    'image_guidance_scale': image_guidance_scale,
                    'method': 'lcm'
                })

    print(f"\nFound {len(eval_csvs)} evaluation results to summarize\n")

    # Prepare combined summary CSV
    summary_rows = []

    # Process each evaluation CSV
    for config in eval_csvs:
        print(f"Processing: gs={config['guidance_scale']}, igs={config['image_guidance_scale']}")

        # Create text summary
        txt_filename = config['filename'].replace('_evaluation_results.csv', '_summary.txt')
        txt_path = os.path.join(args.grid_search_path, txt_filename)

        create_summary_text(
            config['path'],
            txt_path,
            config['guidance_scale'],
            config['image_guidance_scale']
        )
        print(f"  Text summary: {txt_filename}")

        # Calculate averages for CSV
        df = pd.read_csv(config['path'])

        summary_row = {
            'method': config.get('method', 'lcm'),
            'guidance_scale': config['guidance_scale'],
            'image_guidance_scale': config['image_guidance_scale'],
            'num_images': len(df)
        }

        # Calculate mean and std for each metric
        metrics = [col for col in df.columns if col not in ['image_path', 'editing_type']]
        for metric in metrics:
            values = pd.to_numeric(df[metric], errors='coerce')
            summary_row[f'avg_{metric}'] = values.mean()
            summary_row[f'std_{metric}'] = values.std()

        summary_rows.append(summary_row)

    # Create combined summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        # Sort by guidance_scale, then image_guidance_scale
        summary_df = summary_df.sort_values(['guidance_scale', 'image_guidance_scale'])

        summary_df.to_csv(args.output_csv, index=False)
        print(f"\nCombined summary CSV: {args.output_csv}")

        # Print top 10 by CLIP similarity to target
        if 'avg_clip_similarity_target_image' in summary_df.columns:
            print("\n" + "="*60)
            print("Top 10 Parameter Combinations (by CLIP similarity to target):")
            print("="*60)
            top10 = summary_df.nlargest(10, 'avg_clip_similarity_target_image')

            # Select columns to display, including timing if available
            display_cols = ['method', 'guidance_scale', 'image_guidance_scale',
                           'avg_clip_similarity_target_image',
                           'avg_ssim_unedit_part',
                           'avg_lpips_unedit_part']

            if 'avg_elapsed_time_seconds' in summary_df.columns:
                display_cols.extend(['avg_elapsed_time_seconds', 'std_elapsed_time_seconds'])

            print(top10[display_cols].to_string(index=False))

    print("\n" + "="*60)
    print("Summary generation complete!")
    print(f"Text summaries: {args.grid_search_path}/*_summary.txt")
    print(f"Combined CSV: {args.output_csv}")
    print("="*60)


if __name__ == "__main__":
    main()
