import os
import json
import argparse
import re
import glob
import torch

from middleware import GenerationMetricMiddleWare

def get_prompt(dataset):
    prompt_pre = (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant needs to guide the conversation to target topic based on the user's personality and historical conversations.\n\n"
    )
    dialogue_prompt = prompt_pre

    prompts = []
    for data in dataset:
        dialogue_prompt += "The user personas are following:\n\n" + '\n\n'.join(data['personas']) + '\n\n'
        user_prompt = "The topic history is following: " + ' '.join(data['topic_history']) + '\n\n'
        user_prompt += "The target topic is " + data['target_topic'] + '.\n\n'
        user_prompt += "The conversation is follwing:\n\n"
        if len(data['input']) % 2 == 0:
            for j, uttr in enumerate(data['input']):
                if j % 2 == 0:
                    user_prompt += "Assistant: {output}\n\n".format_map({'output': uttr.strip()})
                else:
                    user_prompt += "User: {input}\n\n".format_map({'input': uttr.strip()})
        else:
            for j, uttr in enumerate(data['input']):
                if j % 2 == 0:
                    user_prompt += "User: {output}\n\n".format_map({'output': uttr.strip()})
                else:
                    user_prompt += "Assistant: {input}\n\n".format_map({'input': uttr.strip()})
        user_prompt += "Assistant: "
        full_prompt = dialogue_prompt + user_prompt
        prompts.append(full_prompt)
    return prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default="output/msc_llama-2-7b-chat-hf_no_pos_checkpoint-3000.jsonl")
    parser.add_argument('--data_path', type=str, default="output/ESConv_llama2_7b_causal_kl_checkpoint_429.jsonl")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    data_path_list = glob.glob(args.data_path + "*")
    data_path_list.sort()
    for data_path in data_path_list:
        print('dataset path isL: ' + data_path)
        datas = open(data_path, 'r').readlines()
        datas = [json.loads(data) for data in datas]

        golds = []
        preds = []

        special_chars = [',', '.', '\'', '’', '“', '”', '(', ')', '[', ']', '{', '}', ':', ';', '?', '!', '-', '--']
        for data in datas:
            gold = data['output']['text']
            pred = data['pred_result']
            # pred = data['select_entity']['result']
            for char in special_chars:
                gold = re.sub(rf'([{char}])', r' \1', gold)
                pred = re.sub(rf'([{char}])', r' \1', pred)
            golds.append(gold)
            preds.append(pred)

        metrcis = GenerationMetricMiddleWare()
        bleu = metrcis.compute_bleu(golds, preds, language='en')
        rouge = metrcis.compute_rouge(golds, preds)
        distinct = metrcis.compute_distinct4(preds, language='en')
        print('blue1: {:.4f}; blue2: {:.4f}; rouge-L: {:.4f}; distinct1: {:.4f}; distinct2: {:.4f}'.format(bleu['bleu1'], bleu['bleu2'], rouge['rougeL'], distinct['distinct1'], distinct['distinct2']))