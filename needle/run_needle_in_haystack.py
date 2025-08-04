
"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack
and
https://github.com/nightdessert/Retrieval_Head
"""

# import tiktoken
import os 
import pdb
import glob
import jieba

import json
from transformers import AutoModelForCausalLM, AutoTokenizer,OffloadedCache
import numpy as np
import argparse
from rouge_score import rouge_scorer

import sys
import os

from datetime import datetime, timezone
import time
import torch
import random
import tqdm
import logging
import pprint


from methods.cakekv.cake_cache import CakeprefillKVCache 


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def error_aware_layer_budget_allocation(layer_ratio, total_budget, min_tokens=32, max_tokens=1536):
    importance = np.array(layer_ratio)
    base = np.full_like(importance, min_tokens, dtype=float)
    remaining_budget = total_budget - base.sum()
    extra = np.round(importance * remaining_budget)
    budget_per_layer = base + extra
    budget_per_layer = np.clip(budget_per_layer, min_tokens, max_tokens)
    diff = int(total_budget - budget_per_layer.sum())

    while diff != 0:
        if diff > 0:
            candidates = np.where(budget_per_layer < max_tokens)[0]
            if len(candidates) == 0:
                break
            idx_order = candidates[np.argsort(-importance[candidates])]
            for i in idx_order:
                if budget_per_layer[i] < max_tokens:
                    budget_per_layer[i] += 1
                    diff -= 1
                    if diff == 0:
                        break
        else:
            candidates = np.where(budget_per_layer > min_tokens)[0]
            if len(candidates) == 0:
                break
            idx_order = candidates[np.argsort(importance[candidates])]
            for i in idx_order:
                if budget_per_layer[i] > min_tokens:
                    budget_per_layer[i] -= 1
                    diff += 1
                    if diff == 0:
                        break
    return budget_per_layer.astype(int).tolist()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 args = None,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="needle/PaulGrahamEssays", # PaulGrahamEssays  
                 retrieval_question="What is the best thing to do in San Francisco?", 
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 128000,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "LLaMA",
                 model_path='',
                 model_version=None, 
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True, 
                 step=100, 
                 method='full', 
                 attn_implementation='flash_attention_2',
                 max_capacity_prompts=0
                 ):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param model_path: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.method = method
        self.max_capacity_prompts = max_capacity_prompts
        self.attn_implementation = attn_implementation
        self.save_dir = args.save_dir

        if("/" in model_path):
            self.model_version = model_path.split("/")[-1]
        else: self.model_version = model_path
        

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths

        
        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_path = model_path


        if args.model_path == 'mistralai/Mistral-7B-Instruct-v0.2':
            self.enc = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=args.use_fast_tokenizer,
                padding_side="left",
                revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573'
            )
        else:
            self.enc = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=args.use_fast_tokenizer,
                padding_side="left"
            )
        logging.info("loading from %s" % model_path)
        if args.method == 'snapkv':
            from methods.snapkv.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.method == 'pyramidkv':
            from methods.pyramidkv.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.method == 'cakekv':
            from methods.cakekv.monkeypatch import replace_flashllama_attn_with_cakeattn, replace_flashmistral_attn_with_cakeattn
            replace_flashllama_attn_with_cakeattn()
            replace_flashmistral_attn_with_cakeattn() 
        elif args.method == 'compresskv':
            from methods.compresskv.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.method == 'streamingllm':
            from methods.streamingllm.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.method == "fullkv":
            logging.info(f"using full cache")
        else:
            raise ValueError(f"We does not support {args.method} mode") 

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=args.attn_implementation
        ).eval()
        if args.method != "fullkv":
            window_size = args.window_size
            kernel_size = args.kernel_size
            pooling = args.pooling

            if args.method == "cakekv":
                layers = self.model.model.config.num_hidden_layers
                for i in range(layers):
                    self.model.model.layers[i].self_attn.config.key_size = [args.max_capacity_prompts - window_size]*layers
                    self.model.model.layers[i].self_attn.config.window_size = [window_size]*layers
                    self.model.model.layers[i].self_attn.config.prefill = [True]*layers
                    self.model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
                    self.model.model.layers[i].self_attn.config.tau1 = args.tau1
                    self.model.model.layers[i].self_attn.config.tau2 = args.tau2
                    self.model.model.layers[i].self_attn.config.gamma = args.gamma
                    self.model.model.layers[i].self_attn.config.prefill_cake_evict = [CakeprefillKVCache(
                        cache_size=args.max_capacity_prompts,
                        window_size=window_size,
                        k_seq_dim=2,
                        v_seq_dim=2,
                        num_heads=self.model.model.layers[i].self_attn.num_heads,
                        num_layers=layers,
                        use_cascading=True
                    )]*layers
            else:
                self.model.model.config.window_size = window_size
                self.model.model.config.kernel_size = kernel_size
                self.model.model.config.max_capacity_prompt = args.max_capacity_prompts
                self.model.model.config.pooling = pooling

                if args.method == "pyramidkv":
                    self.model.model.config.pyram_beta = args.pyram_beta
                elif args.method == "compresskv":
                    layers = self.model.model.config.num_hidden_layers
                    layer_score = json.load(open(args.layer_importance_score_path, "r"))["avg_score"]
                    max_capacity_prompt_layer_adaptive = error_aware_layer_budget_allocation(layer_score,args.max_capacity_prompts*layers,32,args.max_capacity_prompts*3)
                    with open(args.importance_head_path, 'r') as f:
                        important_head = json.load(f)
                    important_head = [important_head[str(i)] for i in range(len(important_head))]
                    self.model.model.config.important_heads = important_head
                    self.model.model.config.first_k = args.first_k
                    self.model.model.config.max_capacity_prompt_layer_adaptive = max_capacity_prompt_layer_adaptive

            
    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):
        for context_length in tqdm.tqdm(self.context_lengths, desc=f"Processing the each context length..."):
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                logging.info(f"Context Length: {context_length}, Depth Percent: {depth_percent}")
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                logging.info("result exists, skipping")
                return
            else:
                logging.info("result does not exist, testing")
        # import ipdb;ipdb.set_trace()
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        test_start_time = time.time()

        self.real_needle = "eat a sandwich and sit in Dolores Park on a sunny day"
        # if(self.model_provider in ["LLaMA3", "Mistral"]):

        # prompt = self.enc(prompt, return_tensors="pt")
        input = self.enc(prompt, return_tensors="pt").to(self.model.device)
        context_length = input.input_ids.shape[-1]
        output = self.model.generate(
                **input,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=50,
                pad_token_id=self.enc.eos_token_id,
                )

        output = output[0]
        if args.method == "cakekv":
            layers = len(self.model.model.layers)
            for i in range(layers):
                self.model.model.layers[i].self_attn.config.prefill = [True]*layers
                self.model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
        response = self.enc.decode(output[context_length:], skip_special_tokens=True).strip()
        
        
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        # print(response)
        score = scorer.score(self.needle, response)['rouge1'].recall*100

        results = {
            'model' : self.model_path,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            logging.info (f"-- Test Summary -- ")
            logging.info (f"Duration: {test_elapsed_time:.1f} seconds")
            logging.info (f"Context: {context_length} tokens")
            logging.info (f"Depth: {depth_percent}%")
            logging.info (f"Score: {score}")
            logging.info (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            if not os.path.exists(f'contexts/{self.model_version}'):
                os.makedirs(f'contexts/{self.model_version}')

            with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
        if self.save_results:
            # Save the context to file for retesting
            if self.method == "fullkv":
                results_dir = f'{self.save_dir}/{self.model_version}_{self.method}'
            else:
                results_dir = f'{self.save_dir}/{self.model_version}_{self.method}_{self.max_capacity_prompts}'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            # import ipdb;ipdb.set_trace()
            # Save the result to file for retesting
            p = f'{results_dir}/{context_file_location}_results.json'
            # print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)


    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        if self.method == "fullkv":
            results_dir = f'{self.save_dir}/{self.model_version}_{self.method}'
        else:
            results_dir = f'{self.save_dir}/{self.model_version}_{self.method}_{self.max_capacity_prompts}'
        logging.info("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_path
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            logging.info("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "Phi3", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(context)
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "Phi3", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        logging.info ("\n")
        logging.info ("Starting Needle In A Haystack Testing...")
        logging.info (f"- Model: {self.model_path}")
        logging.info(f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        logging.info (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        logging.info (f"- Needle: {self.needle.strip()}")
        logging.info ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', default=0, metavar='N', type=int)
    parser.add_argument('-e', '--e_len', default=128000, metavar='N', type=int)
    parser.add_argument('--model_path', type=str, default=None, help='name of model')
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "None"])
    parser.add_argument('--model_version', type=str, default=None, help='provider of model')

    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument("--context_length", nargs='+', type=int, 
                        default=[8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000, 72000, 80000, 88000])

    parser.add_argument('--save_dir', type=str, default="needle_results", help='method')
    parser.add_argument('--use_cache', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)


    parser.add_argument('--method', type=str, default=None, help='method')
    parser.add_argument('--max_capacity_prompts', type=int, default=512)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--pooling', type=str, default="avgpool")

    #pyramidkv
    parser.add_argument('--pyram_beta', type=int, default=20, help='pyramidkv parameter')
    #cakekv
    parser.add_argument('--gamma', type=int, default=200)
    parser.add_argument("--tau1", type=float, default=1.0)
    parser.add_argument("--tau2", type=float, default=1.0)

    #compresskv
    parser.add_argument('--layer_importance_score_path',type=str,default=None,help="path to load the layer importance score for budget allocation")
    parser.add_argument('--importance_head_path',type=str,default=None,help="path to load the importance head for selection")
    parser.add_argument('--first_k',type=int,default=4,help="path to load the importance head for selection")
    
    args = parser.parse_args()

    set_seed(args.seed)

    if args.context_length is not None:
        args.context_length = np.array(args.context_length)
    else:
        args.context_length = None

    ht = LLMNeedleHaystackTester(model_path=args.model_path, 
                                model_provider=args.model_provider,
                                model_version=args.model_version, 
                                save_contexts=False,
                                save_results=True,
                                context_lengths=args.context_length,
                                step=args.step, 
                                method=args.method, 
                                max_capacity_prompts=args.max_capacity_prompts,
                                attn_implementation=args.attn_implementation,
                                args=args
                                )

    ht.start_test(args)