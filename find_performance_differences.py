"""
Find Specific Performance Differences

Identifies images where LCM significantly outperformed or underperformed baseline.
"""

import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Find performance differences")
    parser.add_argument('--grid_search_path', type=str, default="output/grid_search",
                       help="Path to grid search output directory")
    parser.add_argument('--lcm_guidance', type=float, default=1.6,
                       help="LCM guidance scale")
    parser.add_argument('--lcm_image_guidance', type=float, default=1.1,
                       help="LCM image guidance scale")
    parser.add_argument('--threshold', type=float, default=0.05,
                       help="Threshold for 'significant' difference")
    parser.add_argument('--top_n', type=int, default=10,
                       help="Number of top examples to show")

    args = parser.parse_args()

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

    print(f"Baseline: {len(baseline_df)} rows")
    print(f"LCM: {len(lcm_df)} rows")
    print(f"Baseline columns: {baseline_df.columns.tolist()}")
    print(f"LCM columns: {lcm_df.columns.tolist()}")
    print(f"\nSample baseline image_path: {baseline_df['image_path'].iloc[0] if len(baseline_df) > 0 else 'N/A'}")
    print(f"Sample LCM image_path: {lcm_df['image_path'].iloc[0] if len(lcm_df) > 0 else 'N/A'}")

    # Merge on image_path
    merged = baseline_df.merge(
        lcm_df,
        on='image_path',
        suffixes=('_baseline', '_lcm')
    )

    print(f"\nMerged: {len(merged)} rows")
    if len(merged) > 0:
        print(f"Sample merged row category: {merged['editing_type_baseline'].iloc[0] if 'editing_type_baseline' in merged.columns else 'N/A'}")
        print(f"Unique categories in merged: {sorted(merged['editing_type_baseline'].unique()) if 'editing_type_baseline' in merged.columns else 'N/A'}")

    # Convert LPIPS to numeric
    merged['lpips_unedit_part_baseline'] = pd.to_numeric(merged['lpips_unedit_part_baseline'], errors='coerce')
    merged['lpips_unedit_part_lcm'] = pd.to_numeric(merged['lpips_unedit_part_lcm'], errors='coerce')

    # Calculate difference (baseline - lcm, positive = LCM better since lower LPIPS is better)
    merged['lpips_improvement'] = merged['lpips_unedit_part_baseline'] - merged['lpips_unedit_part_lcm']

    print("="*100)
    print("CATEGORY 0 (Random Editing): Top Images Where LCM Outperformed Baseline")
    print("="*100)
    print("(Lower LPIPS is better, so positive improvement means LCM is better)")
    print()

    # Category 0: LCM better (largest positive improvement)
    cat0 = merged[merged['editing_type_baseline'] == 0].copy()
    cat0_best = cat0.nlargest(args.top_n, 'lpips_improvement')

    print(f"Total images: {len(cat0)}")
    print(f"Average improvement: {cat0['lpips_improvement'].mean():.6f}")
    print(f"Std deviation: {cat0['lpips_improvement'].std():.6f}")
    print(f"Max improvement: {cat0['lpips_improvement'].max():.6f}")
    print(f"\nTop {args.top_n} images with biggest LCM improvement:\n")
    print(f"{'Image Path':<60} {'Baseline LPIPS':<15} {'LCM LPIPS':<15} {'Difference':<15}")
    print("-"*100)

    for _, row in cat0_best.iterrows():
        print(f"{row['image_path']:<60} {row['lpips_unedit_part_baseline']:<15.6f} "
              f"{row['lpips_unedit_part_lcm']:<15.6f} {row['lpips_improvement']:<15.6f}")

    print("\n\n")
    print("="*100)
    print("CATEGORY 3 (Delete Object): Top Images Where LCM Underperformed Baseline")
    print("="*100)
    print("(Negative difference means LCM is worse)")
    print()

    # Category 3: LCM worse (largest negative improvement = worst performance)
    cat3 = merged[merged['editing_type_baseline'] == 3].copy()
    cat3_worst = cat3.nsmallest(args.top_n, 'lpips_improvement')

    print(f"Total images: {len(cat3)}")
    print(f"Average improvement: {cat3['lpips_improvement'].mean():.6f}")
    print(f"Std deviation: {cat3['lpips_improvement'].std():.6f}")
    print(f"Min improvement (worst): {cat3['lpips_improvement'].min():.6f}")
    print(f"\nTop {args.top_n} images with biggest LCM degradation:\n")
    print(f"{'Image Path':<60} {'Baseline LPIPS':<15} {'LCM LPIPS':<15} {'Difference':<15}")
    print("-"*100)

    for _, row in cat3_worst.iterrows():
        print(f"{row['image_path']:<60} {row['lpips_unedit_part_baseline']:<15.6f} "
              f"{row['lpips_unedit_part_lcm']:<15.6f} {row['lpips_improvement']:<15.6f}")

    print("\n\n")
    print("="*100)
    print("CATEGORY 9 (Change Style): Top Images Where LCM Underperformed Baseline")
    print("="*100)
    print("(Negative difference means LCM is worse)")
    print()

    # Category 9: LCM worse (largest negative improvement = worst performance)
    cat9 = merged[merged['editing_type_baseline'] == 9].copy()
    cat9_worst = cat9.nsmallest(args.top_n, 'lpips_improvement')

    print(f"Total images: {len(cat9)}")
    print(f"Average improvement: {cat9['lpips_improvement'].mean():.6f}")
    print(f"Std deviation: {cat9['lpips_improvement'].std():.6f}")
    print(f"Min improvement (worst): {cat9['lpips_improvement'].min():.6f}")
    print(f"\nTop {args.top_n} images with biggest LCM degradation:\n")
    print(f"{'Image Path':<60} {'Baseline LPIPS':<15} {'LCM LPIPS':<15} {'Difference':<15}")
    print("-"*100)

    for _, row in cat9_worst.iterrows():
        print(f"{row['image_path']:<60} {row['lpips_unedit_part_baseline']:<15.6f} "
              f"{row['lpips_unedit_part_lcm']:<15.6f} {row['lpips_improvement']:<15.6f}")

    # Save to file
    output_file = os.path.join(args.grid_search_path, "performance_differences.txt")
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("Performance Differences: LCM vs Baseline\n")
        f.write("="*100 + "\n\n")

        f.write("CATEGORY 0 (Random Editing): Top Images Where LCM Outperformed Baseline\n")
        f.write("-"*100 + "\n")
        f.write(f"Total: {len(cat0)} images | Avg improvement: {cat0['lpips_improvement'].mean():.6f} | Max: {cat0['lpips_improvement'].max():.6f}\n\n")
        f.write(f"{'Image Path':<60} {'Baseline LPIPS':<15} {'LCM LPIPS':<15} {'Difference':<15}\n")
        f.write("-"*100 + "\n")
        for _, row in cat0_best.iterrows():
            f.write(f"{row['image_path']:<60} {row['lpips_unedit_part_baseline']:<15.6f} "
                   f"{row['lpips_unedit_part_lcm']:<15.6f} {row['lpips_improvement']:<15.6f}\n")

        f.write("\n\n")
        f.write("CATEGORY 3 (Delete Object): Top Images Where LCM Underperformed Baseline\n")
        f.write("-"*100 + "\n")
        f.write(f"Total: {len(cat3)} images | Avg improvement: {cat3['lpips_improvement'].mean():.6f} | Min: {cat3['lpips_improvement'].min():.6f}\n\n")
        f.write(f"{'Image Path':<60} {'Baseline LPIPS':<15} {'LCM LPIPS':<15} {'Difference':<15}\n")
        f.write("-"*100 + "\n")
        for _, row in cat3_worst.iterrows():
            f.write(f"{row['image_path']:<60} {row['lpips_unedit_part_baseline']:<15.6f} "
                   f"{row['lpips_unedit_part_lcm']:<15.6f} {row['lpips_improvement']:<15.6f}\n")

        f.write("\n\n")
        f.write("CATEGORY 9 (Change Style): Top Images Where LCM Underperformed Baseline\n")
        f.write("-"*100 + "\n")
        f.write(f"Total: {len(cat9)} images | Avg improvement: {cat9['lpips_improvement'].mean():.6f} | Min: {cat9['lpips_improvement'].min():.6f}\n\n")
        f.write(f"{'Image Path':<60} {'Baseline LPIPS':<15} {'LCM LPIPS':<15} {'Difference':<15}\n")
        f.write("-"*100 + "\n")
        for _, row in cat9_worst.iterrows():
            f.write(f"{row['image_path']:<60} {row['lpips_unedit_part_baseline']:<15.6f} "
                   f"{row['lpips_unedit_part_lcm']:<15.6f} {row['lpips_improvement']:<15.6f}\n")

    print(f"\n\nResults also saved to: {output_file}")


if __name__ == "__main__":
    main()
