

import json
import os
import glob
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager

#specified the font size
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
    if args.pdf:
        save_path = f"{save_dir}/{path.split('/')[-1]}.pdf"
    else:
        save_path = f"{save_dir}/{path.split('/')[-1]}.png"
    if path.split('/')[-1].lower().endswith('_fullkv'):
        method = "Full_kv"
    else:
        parts = path.split('/')[-1].split('_')
        method = f"{parts[-2].capitalize()}-{parts[-1]}"

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{path}/*.json")

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

    # tick_labels = [f"{x//1000}k" for x in locations]
    if "mistral" in model.lower():
        tick_labels = [f"{x/1000:.1f}k" if x % 1000 != 0 else f"{int(x/1000)}k" for x in locations]
    else:
        tick_labels = [f"{x//1000}k" for x in locations]
    ax.set_xticklabels(tick_labels, rotation=45,fontproperties=prop_13)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop_13)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=38)  

    name_map = {
        'streamingllm': 'StreamingLLM',
        'snapkv': 'SnapKV',
        'pyramidkv': 'PyramidKV',
        'cakekv': 'CAKE',
        'compresskv': 'CompressKV',
        'full_kv': 'FullKV'
    }

    if method.lower() == 'full_kv':
        method_title = name_map['full_kv']
        title_str = rf"$\bf{{{method_title}}}$ Average Score : {df['Score'].mean():.2f}"
    else:
        m = re.match(r'([^-]+)-(\d+)', method, re.IGNORECASE)
        if m:
            method_name, kv_budget = m.groups()
            method_title = name_map.get(method_name.lower(), method_name.capitalize())
            title_str = rf"$\bf{{{method_title}}}$ Average Score : {df['Score'].mean():.2f}"
        else:
            method_title = name_map.get(method.lower(), method.capitalize())
            title_str = rf"$\bf{{{method_title}}}$ Average Score : {df['Score'].mean():.2f}"


    plt.title(title_str,fontproperties=prop_16)
    ax.set_xticklabels(tick_labels, rotation=45,fontproperties=prop_13)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop_13)
    
    if "compresskv" in save_path or "pyramid" in save_path or "streaming" in save_path:
        ax.set_ylabel('', fontproperties=prop_16)
    else:
        plt.ylabel('Depth Percent',fontproperties=prop_16) 
    if "cake" in save_path or "compresskv" in save_path:
        plt.xlabel("Context Length", fontproperties=prop_16)
    else:
        plt.xlabel(' ', fontproperties=prop_16)  
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--save_dir", default="outputs/needle_results_vis", type=str, help="Path to save the output")
    parser.add_argument("--eval_path", default="", type=str, help="Path to save the output")
    parser.add_argument("--pdf", action='store_true', help="save as .pdf")
    parser.add_argument("--expected_answer", type=str, 
                        default="eat a sandwich and sit in Dolores Park on a sunny day.", help="Path to save the output")
    args = parser.parse_args()
    main(args)
