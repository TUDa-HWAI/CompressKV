

import json
import os
import glob
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import os
import re

from matplotlib import font_manager



nimbus_bold_path = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"
prop_bold_16 = font_manager.FontProperties(fname=nimbus_bold_path, size=45)
nimbus_path = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
prop_16 = font_manager.FontProperties(fname=nimbus_path, size=45)

prop_bold_13 = font_manager.FontProperties(fname=nimbus_bold_path, size=38)
prop_13 = font_manager.FontProperties(fname=nimbus_path, size=38)


def round_to_nearest_k(x, k=8000):
    return int(round(x / k)) * k

def main(args):
    
    path = args.eval_path
    save_dir = args.save_dir
    model = args.model.split('/')[-1]
    save_dir = f"{save_dir}/{model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = path.split('/')[-1]
    out_ext = ".pdf" if args.pdf else ".png"
    model_name = model

    save_path = f"{save_dir}/{filename}{out_ext}"

    # Pattern 1: semantic retrieval head
    if "SRH" in path:
        m = re.search(r'top(\d+)', path)
        if m:
            topn = m.group(1)
            save_path = f"{save_dir}/{model_name}_Semantic-Retrieval_mask_top{topn}_heads{out_ext}"
            title = f"Mask out top {topn} Semantic Retrieval Heads"
        else:
            save_path = f"{save_dir}/{model_name}_Semantic-Retrieval_mask_heads{out_ext}"

    # Pattern 2: retrieval head
    elif "block" in path:
        m = re.search(r'top(\d+)', path)
        if m:
            topn = m.group(1)
            save_path = f"{save_dir}/{model_name}_Retrieval_mask_top{topn}_heads{out_ext}"
            title = f"Mask out top {topn} Retrieval Heads"
        else:
            save_path = f"{save_dir}/{model_name}_Retrieval_mask_heads{out_ext}"
    else:
        title = "Without Masking"


    print(save_path)
    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{path}/*.json")
    # import ipdb;ipdb.set_trace()
    data = []
    if "mistral" in model.lower():
        round_k = 500
    if "llama" in model.lower() :
        round_k = 8000

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None)
            if model_response is not None:
                model_response = model_response.lower().split()
            else:
                model_response = []
            needle = json_data.get("needle", None)
            if needle is not None:
                needle = needle.lower()
            else:
                needle = ""
            expected_answer = args.expected_answer.lower().split()
            score = len(set(model_response).intersection(set(expected_answer))) / len(set(expected_answer))*100
            # Round context length to nearest 8k

            context_length_k = round_to_nearest_k(context_length, round_k)
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length_k,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    # Sort context lengths numerically
    locations = sorted(df["Context Length"].unique())
    
    # Pivot with numeric context length
    pivot_table = pd.pivot_table(
        df, 
        values='Score', 
        index=['Document Depth', 'Context Length'], 
        aggfunc='mean'
    ).reset_index()
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    # Create the heatmap
    plt.figure(figsize=(21, 15))
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'shrink': 0.4, 'location': 'right', 'pad':0.02},
        linewidths=0.5,
        linecolor='grey',
        linestyle='--',
        vmin=0, vmax=100,
    )
    ax.set_aspect(0.6) 
    if "mistral" in model.lower():
        tick_labels = [f"{x/1000:.1f}k" if x % 1000 != 0 else f"{int(x/1000)}k" for x in locations]
    else:
        tick_labels = [f"{x//1000}k" for x in locations]

    title_str = f"{title}, Score : {df['Score'].mean():.2f}"

    plt.title(title_str,fontproperties=prop_16)
    ax.set_xticklabels(tick_labels, rotation=45,fontproperties=prop_13)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop_13)
    cbar = ax.collections[0].colorbar

    cbar.ax.tick_params(labelsize=38)  
    plt.xlabel("Context Length", fontproperties=prop_16)
    if "Semantic-Retrieval_mask" in save_path:
        ax.set_ylabel('', fontproperties=prop_16)
    else:
        plt.ylabel('Depth Percent',fontproperties=prop_16) 
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--save_dir", default="../outputs/needle_masking_vis", type=str, help="Path to save the output")
    parser.add_argument("--eval_path", default="", type=str, help="Path to save the output")
    parser.add_argument("--pdf", action='store_true', help="save as .pdf")
    parser.add_argument("--expected_answer", type=str, 
                        default="eat a sandwich and sit in Dolores Park on a sunny day.", help="Path to save the output")
    args = parser.parse_args()
    main(args)
