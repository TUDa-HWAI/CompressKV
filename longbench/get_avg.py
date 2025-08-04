#!/usr/bin/env python3
"""
Compute and normalize per-layer error scores from JSONL files.
Usage:
    python get_avg.py --input_file <path_to_jsonl> --output_file <path_to_output_json>
"""
import argparse
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Average and normalize per-layer reconstruction errors from multiple datasets."
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to the JSONL file containing per-dataset layer error arrays."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to the output JSON file with averaged and normalized layer scores."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    all_values = []
    output_dict = {}
    # Read input JSONL
    with open(args.input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            for dataset, values in data.items():
                all_values.append(values)
                output_dict[dataset] = values

    # Compute mean across datasets (axis=0)
    all_values = np.array(all_values)
    layer_utility = all_values.mean(axis=0)

    # Normalize (L1) across layers
    norm_layer_utility = layer_utility / (layer_utility.sum() + 1e-12)
    output_dict['avg_score'] = norm_layer_utility.tolist()

    # Write output
    with open(args.output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f"Averaged and normalized scores written to {args.output_file}")

if __name__ == '__main__':
    main()