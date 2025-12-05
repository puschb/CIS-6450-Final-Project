"""
Compare Best LCM Run to Baseline

Compares the best LCM parameter combination against standard InstructDiffusion baseline.
"""

import os
import argparse
import pandas as pd


def calculate_improvement(baseline_val, lcm_val, higher_is_better):
    """Calculate percentage improvement."""
    if pd.isna(baseline_val) or pd.isna(lcm_val):
        return "N/A"

    if higher_is_better:
        # For metrics where higher is better
        improvement = ((lcm_val - baseline_val) / baseline_val) * 100
    else:
        # For metrics where lower is better (like LPIPS)
        improvement = ((baseline_val - lcm_val) / baseline_val) * 100

    return improvement


def main():
    parser = argparse.ArgumentParser(description="Compare LCM to baseline")
    parser.add_argument('--grid_search_path', type=str, default="output/grid_search",
                       help="Path to grid search output directory")
    parser.add_argument('--lcm_guidance', type=float, default=1.6,
                       help="LCM guidance scale")
    parser.add_argument('--lcm_image_guidance', type=float, default=1.1,
                       help="LCM image guidance scale")
    parser.add_argument('--baseline_avg_time', type=float, default=1.7526345236345345,
                       help="Baseline average time")
    parser.add_argument('--baseline_std_time', type=float, default=0.036235235,
                       help="Baseline time standard deviation")
    parser.add_argument('--output_txt', type=str, default="output/grid_search/comparison_lcm_vs_baseline.txt",
                       help="Output text file")

    args = parser.parse_args()

    # Key metrics to compare
    METRICS = {
        'ssim_unedit_part': {'name': 'SSIM (Unedited Part)', 'higher_is_better': True},
        'lpips_unedit_part': {'name': 'LPIPS (Unedited Part)', 'higher_is_better': False},
        'clip_similarity_target_image': {'name': 'CLIP Similarity (Target)', 'higher_is_better': True}
    }

    # Category names
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

    # Load LCM results
    lcm_filename = f"instruct-diffusion-lcm_gs{args.lcm_guidance}_igs{args.lcm_image_guidance}_evaluation_results.csv"
    lcm_path = os.path.join(args.grid_search_path, lcm_filename)

    # Load baseline results
    baseline_path = os.path.join(args.grid_search_path, "instruct-diffusion-standard_evaluation_results.csv")

    if not os.path.exists(lcm_path):
        print(f"Error: LCM results not found at {lcm_path}")
        return

    if not os.path.exists(baseline_path):
        print(f"Error: Baseline results not found at {baseline_path}")
        return

    print("Loading results...")
    lcm_df = pd.read_csv(lcm_path)
    baseline_df = pd.read_csv(baseline_path)

    # Calculate overall statistics
    with open(args.output_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LCM vs Baseline Comparison\n")
        f.write("="*80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  LCM:      guidance={args.lcm_guidance}, image_guidance={args.lcm_image_guidance}, steps=4\n")
        f.write(f"  Baseline: guidance=5.0, image_guidance=1.25, steps=50\n")
        f.write("\n")

        # Overall comparison
        f.write("="*80 + "\n")
        f.write("Overall Performance Comparison\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Metric':<35} {'Baseline':<15} {'LCM':<15} {'Improvement':<15}\n")
        f.write("-"*80 + "\n")

        for metric_key, metric_info in METRICS.items():
            baseline_vals = pd.to_numeric(baseline_df[metric_key], errors='coerce')
            lcm_vals = pd.to_numeric(lcm_df[metric_key], errors='coerce')

            baseline_mean = baseline_vals.mean()
            baseline_std = baseline_vals.std()
            lcm_mean = lcm_vals.mean()
            lcm_std = lcm_vals.std()

            improvement = calculate_improvement(baseline_mean, lcm_mean, metric_info['higher_is_better'])

            baseline_str = f"{baseline_mean:.4f}±{baseline_std:.4f}"
            lcm_str = f"{lcm_mean:.4f}±{lcm_std:.4f}"

            if improvement == "N/A":
                improvement_str = "N/A"
            else:
                sign = "+" if improvement > 0 else ""
                improvement_str = f"{sign}{improvement:.2f}%"

            f.write(f"{metric_info['name']:<35} {baseline_str:<15} {lcm_str:<15} {improvement_str:<15}\n")

        # Timing comparison
        f.write("\n")
        f.write(f"{'Metric':<35} {'Baseline':<15} {'LCM':<15} {'Speedup':<15}\n")
        f.write("-"*80 + "\n")

        lcm_time_vals = pd.to_numeric(lcm_df['elapsed_time_seconds'], errors='coerce')
        lcm_time_mean = lcm_time_vals.mean()
        lcm_time_std = lcm_time_vals.std()

        speedup = args.baseline_avg_time / lcm_time_mean if lcm_time_mean > 0 else float('inf')

        baseline_time_str = f"{args.baseline_avg_time:.4f}±{args.baseline_std_time:.4f}"
        lcm_time_str = f"{lcm_time_mean:.4f}±{lcm_time_std:.4f}"
        speedup_str = f"{speedup:.2f}x faster"

        f.write(f"{'Inference Time (seconds)':<35} {baseline_time_str:<15} {lcm_time_str:<15} {speedup_str:<15}\n")

        # Per-category comparison
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("Per-Category Performance Comparison\n")
        f.write("="*80 + "\n\n")

        for category_id in sorted(baseline_df['editing_type'].unique()):
            category_name = category_names.get(str(category_id), f"Category {category_id}")

            baseline_cat = baseline_df[baseline_df['editing_type'] == category_id]
            lcm_cat = lcm_df[lcm_df['editing_type'] == category_id]

            f.write(f"Category {category_id}: {category_name}\n")
            f.write(f"  Images: {len(baseline_cat)}\n")
            f.write("-"*80 + "\n")

            f.write(f"  {'Metric':<33} {'Baseline':<15} {'LCM':<15} {'Improvement':<15}\n")
            f.write("  " + "-"*76 + "\n")

            for metric_key, metric_info in METRICS.items():
                baseline_vals = pd.to_numeric(baseline_cat[metric_key], errors='coerce')
                lcm_vals = pd.to_numeric(lcm_cat[metric_key], errors='coerce')

                baseline_mean = baseline_vals.mean()
                lcm_mean = lcm_vals.mean()

                improvement = calculate_improvement(baseline_mean, lcm_mean, metric_info['higher_is_better'])

                baseline_str = f"{baseline_mean:.4f}" if not pd.isna(baseline_mean) else "nan"
                lcm_str = f"{lcm_mean:.4f}" if not pd.isna(lcm_mean) else "nan"

                if improvement == "N/A":
                    improvement_str = "N/A"
                else:
                    sign = "+" if improvement > 0 else ""
                    improvement_str = f"{sign}{improvement:.2f}%"

                f.write(f"  {metric_info['name']:<33} {baseline_str:<15} {lcm_str:<15} {improvement_str:<15}\n")

            f.write("\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("Summary\n")
        f.write("="*80 + "\n")
        f.write(f"• Speedup: {speedup:.2f}x faster ({args.baseline_avg_time:.3f}s → {lcm_time_mean:.3f}s)\n")

        # Calculate overall quality change
        for metric_key, metric_info in METRICS.items():
            baseline_vals = pd.to_numeric(baseline_df[metric_key], errors='coerce')
            lcm_vals = pd.to_numeric(lcm_df[metric_key], errors='coerce')
            improvement = calculate_improvement(baseline_vals.mean(), lcm_vals.mean(), metric_info['higher_is_better'])

            if improvement != "N/A":
                direction = "improvement" if improvement > 0 else "degradation"
                f.write(f"• {metric_info['name']}: {abs(improvement):.2f}% {direction}\n")

        f.write("="*80 + "\n")

    print(f"\nComparison complete!")
    print(f"Results saved to: {args.output_txt}")


if __name__ == "__main__":
    main()
