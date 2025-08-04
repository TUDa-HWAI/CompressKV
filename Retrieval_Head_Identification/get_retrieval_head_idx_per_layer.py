
import ipdb;ipdb.set_trace()
import json
import numpy as np
import argparse

def main(input_path: str, output_path: str):
    with open(input_path) as file:
        head_list = json.loads(file.readline())
    head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
    # head_score_list -> [[[head_idx, layer_idx], retrieval_score],...]
    # get layer_wise retrieval head index based on the average retrieval score
    layer_wise_retrieval_head_idx = {}
    for head_score in head_score_list:
        # head_idx, layer_idx = head_score[0]
        layer_idx, head_idx = head_score[0]
        if layer_idx not in layer_wise_retrieval_head_idx:
            layer_wise_retrieval_head_idx[layer_idx] = []
        layer_wise_retrieval_head_idx[layer_idx].append([head_idx, head_score[1]])

    # sort the retrieval head index based on the retrieval score
    for layer_idx in layer_wise_retrieval_head_idx:
        layer_wise_retrieval_head_idx[layer_idx] = sorted(layer_wise_retrieval_head_idx[layer_idx], key=lambda x: x[1], reverse=True)
        
    # no score, only head index
    layer_wise_retrieval_head_idx_no_score = {}
    for layer_idx in layer_wise_retrieval_head_idx:
        layer_wise_retrieval_head_idx_no_score[layer_idx] = [l[0] for l in layer_wise_retrieval_head_idx[layer_idx]] 

    with open(output_path, 'w') as f_out:
        json.dump(layer_wise_retrieval_head_idx_no_score, f_out, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='get_retrieval_head_idx_per_layer.py',
        description='Convert head scores JSON to layer-wise head-index JSON'
    )
    parser.add_argument(
        'input_path',
        nargs='?',
        default='head_score/Llama-3.1-8B-Instruct_SRH.json',
        help='Path to the input JSON file'
    )
    parser.add_argument(
        'output_path',
        nargs='?',
        default='head_test.json',
        help='Path to write the output JSON file'
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)