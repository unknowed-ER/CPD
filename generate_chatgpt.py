import json
import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import argparse
import warnings
import os
import prompt
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

'''ddp'''
import torch.distributed as dist
import torch.multiprocessing as mp

import requests
import json
import threading


def collate_fn(batch):
    prompt, input, output, input_ids, attention_mask, split_tokens = [], [], [], [], [], []
    for data in batch:
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])
        split_tokens.append(data['split_token'])
        prompt.append(data['prompt'])
        input.append(data['input'])
        output.append(data['output'])
    # left padding
    max_input_len = max([len(input_id) for input_id in input_ids])
    for i in range(len(batch)):
        input_ids[i] = [0] * (max_input_len-len(input_ids[i])) + input_ids[i]
        attention_mask[i] = [0] * (max_input_len-len(attention_mask[i])) + attention_mask[i]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    return {
        "prompt": prompt,
        "input": input,
        "output": output,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'split_tokens': split_tokens
    }



@torch.inference_mode()
def evaluate_data(
        input,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.3,
):
    apiUrl = 'http://222.20.76.130:8000'
    apiKey = 'sk-'
    # apiKey = 'sk-aVx12m1Sm1UaoboV66B43aBa93984dF8970b09CdEa0d294d'

    model = 'gpt-3.5-turbo'
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey
    }

    # post = {
    # 'model': model,
    # "prompt": "log,logo： ,",
    # "n": 1,
    # "size": "1024x1024"
    # }
    post = {
        'model': model,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'repetition_penalty': repetition_penalty,
        'messages': [
            {
                'role': 'user',
                'content': input['prompt']
            }
        ]
    }

    try:
        response = requests.post(
            apiUrl,
            headers=headers,
            data=json.dumps(post),
            timeout=120
        )
        if response.status_code == 200:
            if response.text:
                json_data = response.json()
            print(json_data['choices'][0]['message']['content'])
        else:
            print(f"Request failed, response content: {response.text}")
    except Exception as e:
        print('Error:', e)
    return json_data['choices'][0]['message']['content']


def preprocess_gen(data_point):
    user_prompt = "The following is a conversation between two people called Speaker 1 and a Speaker 2. " \
                              "You needs to generate response **briefly** and **precisely** according to the given dialog history and user personas.\n"
    user_prompt += data_point['instruction']
    user_prompt += 'Dialogue History:\n'
    user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
    user_prompt += data_point["output"]['user'] + ' : '
    return {
        "prompt": user_prompt,
        "input": data_point["input"],
        "output": data_point["output"],
        "split_token": data_point["output"]['user'] + ' : '
    }


# python generate_chatgpt.py --data_path datasets/ESConv/ESConv_test.jsonl --output_path output/ESConv_chatgpt.jsonl
# python generate_chatgpt.py --data_path datasets/msc/msc_test.jsonl --output_path output/msc_chatgpt.jsonl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets/ESConv/ESConv_test.jsonl")
    parser.add_argument("--output_path", type=str, default="output/ESConv_chatgpt.jsonl")
    parser.add_argument("--cutoff_len", type=int, default=4096, help='')
    parser.add_argument("--num_beams", type=int, default=1, help='beam_search depth, 1do_sample')
    parser.add_argument("--max_new_tokens", type=int, default=256, help='')
    parser.add_argument("--use_local", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    args = parser.parse_args()

    print(args)

    datasets = load_dataset('json', data_files=args.data_path)['train']
    if 'msc' in args.data_path:
        datasets = Dataset.from_dict(datasets.shuffle()[:2000])
    with open(args.output_path, 'a') as f:
        for input_data in tqdm(datasets):
            data = preprocess_gen(input_data)
            result = evaluate_data(data)
            output_data = {
                'input': input_data['input'],
                'output': input_data['output'],
                # 'entities': input_data['entities'],  # mscESConv，entity。
                'pred_result': result,
            }
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')