"""
Timing measurement script for Phase 1 (baseline) vs Phase 2 (adaptive).
Measures processing time on a sample of images to validate speedup.
"""

import json
import time
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

# Import both implementations
import run_editing_instructpix2pix as baseline
import run_editing_instructpix2pix_adaptive as adaptive


def measure_timing(num_images=20):
    """
    Measure timing for Phase 1 (baseline) vs Phase 2 (adaptive).
    
    Args:
        num_images: Number of images to test (default: 20)
    
    Returns:
        dict: Timing results including speedup percentage
    """
    # Load mapping file
    with open("data/mapping_file.json", "r") as f:
        mapping_dict = json.load(f)
    
    data_list = list(mapping_dict.values())[:num_images]
    
    print("=" * 70)
    print(f"TIMING TEST: Phase 1 vs Phase 2 ({len(data_list)} images)")
    print("=" * 70)
    
    # Load model once
    print("\nLoading model...")
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    print("✓ Model loaded\n")
    
    # Set model globally for both modules
    baseline.model = model
    adaptive.model = model
    
    # Phase 1: Baseline (Fixed 50 steps)
    print("Running Phase 1 (Baseline - Fixed 50 steps)...")
    start1 = time.time()
    for i, data in enumerate(data_list):
        print(f"  [{i+1}/{len(data_list)}] Processing...")
        image_filename = data['image_path'].split('/')[-1]
        baseline.edit_instruct_pix2pix(
            f"data/annotation_images/{image_filename}",
            data['editing_prompt'],
            data['editing_instruction'],
            f"output/timing_test/phase1/{image_filename}"
        )
    time1 = time.time() - start1
    print(f"✓ Phase 1 complete: {time1:.1f}s\n")
    
    # Phase 2: Adaptive scheduling
    print("Running Phase 2 (Adaptive scheduling)...")
    start2 = time.time()
    for i, data in enumerate(data_list):
        print(f"  [{i+1}/{len(data_list)}] Processing...")
        image_filename = data['image_path'].split('/')[-1]
        adaptive.edit_instruct_pix2pix(
            f"data/annotation_images/{image_filename}",
            data['original_prompt'],
            data['editing_prompt'],
            data['editing_instruction'],
            int(data['editing_type_id']),
            f"output/timing_test/phase2/{image_filename}",
            use_adaptive=True
        )
    time2 = time.time() - start2
    print(f"✓ Phase 2 complete: {time2:.1f}s\n")
    
    # Calculate results
    speedup = ((time1 - time2) / time1 * 100)
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nPhase 1 (Baseline):")
    print(f"  Total time: {time1:.1f}s ({time1/len(data_list):.2f}s per image)")
    print(f"\nPhase 2 (Adaptive):")
    print(f"  Total time: {time2:.1f}s ({time2/len(data_list):.2f}s per image)")
    print(f"\nSPEEDUP: {speedup:+.1f}%")
    print(f"Time saved: {time1-time2:.1f}s")
    print("=" * 70)
    
    results = {
        "num_images": len(data_list),
        "phase1_total_seconds": round(time1, 1),
        "phase1_per_image": round(time1/len(data_list), 2),
        "phase2_total_seconds": round(time2, 1),
        "phase2_per_image": round(time2/len(data_list), 2),
        "speedup_percent": round(speedup, 1),
        "time_saved_seconds": round(time1 - time2, 1)
    }
    
    return results


if __name__ == "__main__":
    results = measure_timing(num_images=20)
    
    # Save results to JSON
    with open("timing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to timing_results.json")
