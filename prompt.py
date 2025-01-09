import itertools

import transformers
import copy
from torch.nn.utils.rnn import pad_sequence
import torch
import random

from tqdm import tqdm
import json
from transformers import Conversation

class prompt:
    def __init__(self, tokenizer, max_len, add_eos=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos=add_eos

    def preprocess_gen(self, data_point):
        user_prompt = data_point['input']
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids

    def postprocess(self, text, render=True):
        output = text.replace('�','').replace('<s>','')
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

class instruct_prompt(prompt):
    prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    prompt_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n\n### Input:{input}\n\n### Response:"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"


    def preprocess_gen(self, data_point):
        if 'history' not in data_point:
        # single instruction format {'instruction':..,'input':..}
            if 'input' in data_point:
                user_prompt = self.prompt_input.format_map(data_point)
            else:
                user_prompt = self.prompt.format_map(data_point)
        else:
            user_prompt = ''
            lens = len(data_point['history'])
            for i in range(lens):
                user_prompt += self.prompt_history.format_map(data_point['history'][i])
            user_prompt += self.prompt_post.format_map({'input':data_point['input']})
        user_prompt=self.prompt.format_map({'instruction':user_prompt})
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids

    def preprocess_train(self, data_point):
        # single instruction format {'instruction':..,'input':..,'output':..}
        if 'instruction' in data_point:
            if 'input' in data_point:
                user_prompt = self.prompt_input.format_map(data_point)
            else:
                user_prompt = self.prompt.format_map(data_point)
            output = data_point["output"]
        # multi turn format {'input':[..], 'output':[..]}
        else:
            user_prompt = ''
            lens = len(data_point['input'])
            for i in range(lens-1):
                user_prompt += self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1]})
            user_prompt = self.prompt.format_map({'instruction': user_prompt})
            output = data_point['output'][-1]

        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len + 1,
        )["input_ids"])- 1  # no eos token
        full_tokens = self.tokenizer(
            user_prompt + output,
            truncation=True,
            max_length=self.max_len + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def preprocess_text(self, data_point):
        if data_point['input'] != '':
            user_prompt = self.prompt_input.format_map(data_point)
        else:
            user_prompt = self.prompt.format_map(data_point)
        return {
            'input':user_prompt,
            'output':data_point['output'],
        }

    def data_collator(self,):
        return transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def postprocess(self, text, render=True):
        #import pdb;pdb.set_trace()
        output = text.split("### Response:")[1].strip()
        output = output.replace("Belle", "Vicuna")
        printf(text)
        output = text.split("### Response:")[1].strip()
        output = output.replace("Belle", "Vicuna")
        printf(output)
        if '###' in output:
            output = output.split("###")[0]
        if 'User' in output:
            output = output.split("User")[0]
        # output = output.replace('�','')
        output = output.replace('�','').replace('</s>', '') 
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

class chat_prompt(prompt):
    prompt_pre = (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant is intelligent, knowledgeable and polite to answer questions of user.\n\n"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def preprocess_text(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})
        return {
            'input':user_prompt,
            'output':data_point['output'][-1].strip(),
        }

    def preprocess_gen(self, data_point):
        # ，prompt
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})
        inputs = self.tokenizer(
                user_prompt,
                truncation=True,
                padding=False,
                max_length=self.max_len+1,
            )
        return {"input_ids": inputs["input_ids"][:-1]} # delete eos

    def preprocess_train(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})

        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1 # remove extra eos
        if isinstance(self.tokenizer, transformers.GPT2TokenizerFast):
            len_user_prompt_tokens += 1 # gpt will not add eos
            full_prompt = user_prompt + data_point["output"][-1].strip() + self.tokenizer.eos_token
        else:
            full_prompt = user_prompt + data_point["output"][-1].strip()
        if self.add_eos:
            full_tokens = self.tokenizer(
                full_prompt,
                truncation=True,
                padding=False,
                max_length=self.max_len,
            )["input_ids"] # need eos
        else:
            full_tokens = self.tokenizer(
                full_prompt,
                truncation=True,
                padding=False,
                max_length=self.max_len+1,
            )["input_ids"][:-1] # delete eos
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def data_collator(self,):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer)
    def postprocess(self, text, render=False, split=True):
        output = text.split("Assistant:")[-1]
        if split and 'User:' in output:
            output = output.split("User:")[0]
        output = output.replace('�','') 
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def get_data_collator(self):
        return transformers.DataCollatorForLanguageModeling

    def preprocess_train1(self, data_point):
        user_prompt = self.prompt_pre    
        len_pre = len(self.tokenizer(
            user_prompt,
            padding=False,
            add_special_tokens=False,
        )["input_ids"])
        labels = [-100] * (len_pre+1) # add bos

        lens = len(data_point['input'])
        for i in range(lens):
            prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            user_prompt += prompt
            ids = self.tokenizer(
                prompt,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            len1 = len(self.tokenizer(
                f"User:{data_point['input'][i]}\n\nAssistant:",
                padding=False,
                add_special_tokens=False,
            )["input_ids"])
            label = [-100]*len1+ids[len1:-2]+[self.tokenizer.eos_token_id, -100] # \n\n
            labels += label
        input_ids = self.tokenizer(
            user_prompt[:-2], # delete last two \n\n
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        labels = (labels[:-2]+[self.tokenizer.eos_token_id])[:self.max_len] # user

        if labels[-1]!= -100:
            labels[-1] = self.tokenizer.eos_token_id

        assert labels != len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * (len(input_ids)),
        }

    def preprocess_split(self, data_point, drop_single=False):

        user_prompt = self.prompt_pre
        len_pre = len(self.tokenizer(
            user_prompt,
            add_special_tokens=False,
        ))
        assert len_pre < self.max_len

        tokenized_lens = []
        for i in range(len(data_point['input'])):
            tmp_prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            single_len =(len(self.tokenizer(
                tmp_prompt,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]))
            # ：/2 inputoutput
            while single_len > self.max_len:
                tmp_len1 = len(data_point['input'][i])
                tmp_len2 = len(data_point['output'][i])
                if tmp_len2 > tmp_len1:
                    data_point['output'][i] = data_point['output'][i][:tmp_len2//2]
                else:
                    data_point['input'][i] = data_point['input'][i][:tmp_len1//2]
                prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
                single_len =(len(self.tokenizer(
                    prompt,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]))
            tokenized_lens.append(single_len)

        num_tokens = len_pre
        left, right = 0,0
        new_turns = []
        while right < len(tokenized_lens):
            
            l = tokenized_lens[right]
            num_tokens += l

            if num_tokens > self.max_len:
                if left == right:
                    right += 1
                new_turns.append({
                    'input': data_point['input'][left:right],
                    'output': data_point['output'][left:right],
                })
                left = right
                num_tokens = len_pre
            else:
                right +=1
        if right > left:
            new_turns.append({
                'input': data_point['input'][left:right],
                'output': data_point['output'][left:right],
            })
        if drop_single:
            new_turns = [d for d in new_turns if len(d['input'])>1]
        if len(new_turns) > 1:
            print(sum(tokenized_lens)+len_pre,[len(new_turns[i]['input']) for i in range(len(new_turns))])
        return new_turns


class chat_topic_prompt(chat_prompt):
    dataset = None
    prompt_pre = (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant needs to guide the conversation to target topic based on the user's personality and historical conversations.\n\n"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def set_dataset(self, dataset, vocab):
        self.dataset = dataset
        self.vocab = vocab
    def preprocess_train(self, conversation):
        dialogue_prompt = self.prompt_pre
        user_id = int(conversation[0])
        user_persona_ids = self.vocab.user_to_Sentidx[str(user_id)]
        user_personas = [self.vocab.idx_to_userSent[i] for i in user_persona_ids]
        dialogue_prompt += "The user personas are following:\n\n" + '\n\n'.join(user_personas) + '\n\n'
        contexts = conversation[-3]
        conv_id = conversation[-1]
        utterances = conversation[1:-3]
        uttr_len = len(utterances)
        if self.dataset == 'TG-ReDial':
            skip_len = 2
            for i in range(len(contexts)):
                contexts[i] = ''.join(contexts[i])
            for i in range(uttr_len):
                utterances[i][0] = ''.join(utterances[i][0])
        elif self.dataset == 'PersonaChat':
            skip_len = 1
            for i in range(len(contexts)):
                contexts[i] = ' '.join(contexts[i])
            for i in range(uttr_len):
                utterances[i][0] = ' '.join(utterances[i][0])
        processed_data = []
        for i in range(2, uttr_len, skip_len):
            if self.dataset == 'PersonaChat' and (utterances[i - 1][2][-1] == '[UNK]' or utterances[i][2][-1] == '[UNK]'):
                continue  # pad
            response = utterances[i]
            if len(response[2]) == 0:
                continue
            action_R = response[2]
            resp = response[0]
            context = contexts[:i]
            if response[1][-1] is None:
                response[1][-1] = '[UNK]'
            user_prompt = "The topic history is following: " + ' '.join(response[1][:-1]) + '\n\n'
            user_prompt += "The target topic is " + response[1][-1] + '.\n\n'
            user_prompt += "The conversation is follwing:\n\n"

            if i % 2 == 0:
                for j, uttr in enumerate(context):
                    if j % 2 == 0:
                        user_prompt += "Assistant: {output}\n\n".format_map({'output':uttr.strip()})
                    else:
                        user_prompt += "User: {input}\n\n".format_map({'input': uttr.strip()})
            else:
                for j, uttr in enumerate(context):
                    if j % 2 == 0:
                        user_prompt += "User: {output}\n\n".format_map({'output':uttr.strip()})
                    else:
                        user_prompt += "Assistant: {input}\n\n".format_map({'input': uttr.strip()})
            user_prompt += "Assistant: "
            len_user_prompt_tokens = len(self.tokenizer(
                dialogue_prompt + user_prompt,
                truncation=True,
                max_length=self.max_len,
            )["input_ids"]) - 1 # remove extra eos
            if self.add_eos:
                if isinstance(self.tokenizer, transformers.GPT2TokenizerFast):
                    len_user_prompt_tokens += 1 # gpt will not add eos
                    full_prompt = dialogue_prompt + user_prompt + resp.strip() + self.tokenizer.eos_token
                else:
                    full_prompt = dialogue_prompt + user_prompt + resp.strip()
            else:
                if isinstance(self.tokenizer, transformers.GPT2TokenizerFast):
                    len_user_prompt_tokens += 1 # gpt will not add eos
                    full_prompt = dialogue_prompt + user_prompt + self.tokenizer.eos_token
                else:
                    full_prompt = dialogue_prompt + user_prompt
            if self.add_eos:
                full_tokens = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    padding=False,
                    max_length=self.max_len,
                )["input_ids"]  # need eos
            else:
                full_tokens = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    padding=False,
                    max_length=self.max_len+1,
                )["input_ids"][:-1]  # delete eos
            processed_data.append({
                "input_ids": full_tokens,
                "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
                "attention_mask": [1] * (len(full_tokens)),
                "input": context,
                'output': response[0],
                'personas': user_personas,
                'topic_history': response[1][:-1],
                'gold_topic': response[2][1],
                'target_topic': response[1][-1]
            })
        return processed_data

    def preprocess_dataset(self, dataset):
        from tqdm import tqdm
        processed_dataset = []
        for data in tqdm(dataset):
            processed_data = self.preprocess_train(data)
            processed_dataset += processed_data
        return processed_dataset

class chat_topic_prompt(chat_prompt):
    dataset = None
    prompt_pre = (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant needs to guide the conversation to target topic based on the user's personality and historical conversations.\n\n"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def set_dataset(self, dataset, vocab):
        self.dataset = dataset
        self.vocab = vocab
    def preprocess_train(self, conversation):
        dialogue_prompt = self.prompt_pre
        user_id = int(conversation[0])
        user_persona_ids = self.vocab.user_to_Sentidx[str(user_id)]
        user_personas = [self.vocab.idx_to_userSent[i] for i in user_persona_ids]
        dialogue_prompt += "The user personas are following:\n\n" + '\n\n'.join(user_personas) + '\n\n'
        contexts = conversation[-3]
        conv_id = conversation[-1]
        utterances = conversation[1:-3]
        uttr_len = len(utterances)
        if self.dataset == 'TG-ReDial':
            skip_len = 2
            for i in range(len(contexts)):
                contexts[i] = ''.join(contexts[i])
            for i in range(uttr_len):
                utterances[i][0] = ''.join(utterances[i][0])
        elif self.dataset == 'PersonaChat':
            skip_len = 1
            for i in range(len(contexts)):
                contexts[i] = ' '.join(contexts[i])
            for i in range(uttr_len):
                utterances[i][0] = ' '.join(utterances[i][0])
        processed_data = []
        for i in range(2, uttr_len, skip_len):
            if self.dataset == 'PersonaChat' and (utterances[i - 1][2][-1] == '[UNK]' or utterances[i][2][-1] == '[UNK]'):
                continue  # pad
            response = utterances[i]
            if len(response[2]) == 0:
                continue
            action_R = response[2]
            resp = response[0]
            context = contexts[:i]
            if response[1][-1] is None:
                response[1][-1] = '[UNK]'
            user_prompt = "The topic history is following: " + ' '.join(response[1][:-1]) + '\n\n'
            user_prompt += "The target topic is " + response[1][-1] + '.\n\n'
            user_prompt += "The conversation is follwing:\n\n"

            if i % 2 == 0:
                for j, uttr in enumerate(context):
                    if j % 2 == 0:
                        user_prompt += "Assistant: {output}\n\n".format_map({'output':uttr.strip()})
                    else:
                        user_prompt += "User: {input}\n\n".format_map({'input': uttr.strip()})
            else:
                for j, uttr in enumerate(context):
                    if j % 2 == 0:
                        user_prompt += "User: {output}\n\n".format_map({'output':uttr.strip()})
                    else:
                        user_prompt += "Assistant: {input}\n\n".format_map({'input': uttr.strip()})
            user_prompt += "Assistant: "
            len_user_prompt_tokens = len(self.tokenizer(
                dialogue_prompt + user_prompt,
                truncation=True,
                max_length=self.max_len,
            )["input_ids"]) - 1 # remove extra eos
            if self.add_eos:
                if isinstance(self.tokenizer, transformers.GPT2TokenizerFast):
                    len_user_prompt_tokens += 1 # gpt will not add eos
                    full_prompt = dialogue_prompt + user_prompt + resp.strip() + self.tokenizer.eos_token
                else:
                    full_prompt = dialogue_prompt + user_prompt + resp.strip()
            else:
                if isinstance(self.tokenizer, transformers.GPT2TokenizerFast):
                    len_user_prompt_tokens += 1 # gpt will not add eos
                    full_prompt = dialogue_prompt + user_prompt + self.tokenizer.eos_token
                else:
                    full_prompt = dialogue_prompt + user_prompt
            if self.add_eos:
                full_tokens = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    padding=False,
                    max_length=self.max_len,
                )["input_ids"]  # need eos
            else:
                full_tokens = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    padding=False,
                    max_length=self.max_len+1,
                )["input_ids"][:-1]  # delete eos
            processed_data.append({
                "input_ids": full_tokens,
                "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
                "attention_mask": [1] * (len(full_tokens)),
                "input": context,
                'output': response[0],
                'personas': user_personas,
                'topic_history': response[1][:-1],
                'gold_topic': response[2][1],
                'target_topic': response[1][-1]
            })
        return processed_data

    def preprocess_dataset(self, dataset):
        from tqdm import tqdm
        processed_dataset = []
        for data in tqdm(dataset):
            processed_data = self.preprocess_train(data)
            processed_dataset += processed_data
        return processed_dataset

class persona_prompt(chat_prompt):
    prompt_pre = (
        "The following is a conversation between two people called Speaker 1 and a Speaker 2. "
        "You needs to generate response **briefly** and **precisely** according to the given dialog history and user personas.\n"
    )
    # prompt_pre = (
    #     "The following is a conversation between an AI assistant called Assistant and a human user called User. "
    #     "The assistant needs to guide the conversation to target topic based on the user's personality and historical conversations.\n\n"
    # )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def __init__(self, tokenizer, max_len, add_eos=True, drop=0, use_data_instruction=True, is_persona=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos = add_eos
        self.drop = drop
        self.use_data_instruction = use_data_instruction
        if use_data_instruction:
            self.prompt_pre = "The following is a conversation between two people called Speaker 1 and a Speaker 2. " \
                              "You needs to generate response **briefly** and **precisely** according to the given dialog history and user personas.\n"
        else:
            self.prompt_pre = "The following is a conversation between two people called Speaker 1 and a Speaker 2. " \
                              "You needs to generate response **briefly** and **precisely** according to the given dialog history.\n"
        self.is_persona = is_persona
        if not self.is_persona:
            self.prompt_pre = "The following is a conversation between an AI assistant called Assistant and a human user called User. "\
                              "The assistant needs to generate response **briefly** and **precisely** according to the given dialog history.\n\n"

    def collect_fn(self, batch_datas):
        return batch_datas

    def preprocess_gen(self, data_point):
        user_prompt = self.prompt_pre
        user_prompt += data_point['instruction']
        user_prompt += 'Dialogue History:\n'
        if 'lama' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
            user_prompt += data_point["output"]['user'] + ' : '
        elif 'Qwen' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
            user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : ' + '<|im_end|>'
        full_input = self.tokenizer(user_prompt, truncation=True, padding=False, max_length=self.max_len)
        return {
            "prompt": user_prompt,
            "input": data_point["input"],
            "output": data_point["output"],
            "input_ids": full_input["input_ids"][:-1],  # EOS'</s>', token_ids=2
            "attention_mask": full_input['attention_mask'][:-1],
            "split_token": data_point["output"]['user'] + ' : '
        }

    def preprocess_gen_conflict(self, data_point):
        user_prompt = self.prompt_pre
        user_prompt += data_point['instruction']
        user_prompt += 'Dialogue History:\n'
        user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input'][:-1]]) + '\n'
        user_prompt += data_point['input'][-1]['user'] + ' : '
        # user_prompt += data_point["output"]['user'] + ' : '
        full_input = self.tokenizer(user_prompt, truncation=True, padding=False, max_length=self.max_len)
        return {
            "prompt": user_prompt,
            "input": data_point["input"],
            "output": data_point["output"],
            "input_ids": full_input["input_ids"][:-1],
            "attention_mask": full_input['attention_mask'][:-1],
            "split_token": data_point["output"]['user'] + ' : '
        }

    def get_entity_turns(self, data_point):
        select_entity = data_point['select_entity']
        for j in range(len(select_entity)):
            for i in range(len(data_point['input'])-1, -1, -1):
                if select_entity[j]['text'] == data_point['input'][i]['text']:
                    data_point['select_entity'][j]['turn_id'] = len(data_point['input'])-1-i
                    break
        return data_point

    def statistic(self, dataset):
        causal_dist_num = [0 for _ in range(150)]
        all_context = set()
        for data in tqdm(dataset):
            for uttr in data['input']:
                all_context.add(uttr['text'])
            select_entity = data['select_entity']
            for entity in select_entity:
                causal_dist_num[entity['turn_id']] += 1
        self.causal_dist_num = causal_dist_num
        self.all_context = list(all_context)

    def preprocess_train(self, data_point):
        user_prompt = self.prompt_pre
        if self.use_data_instruction:
            # ppl
            user_prompt += data_point['instruction']
            user_prompt += 'Dialogue History:\n'
        if 'lama' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
            user_prompt += data_point["output"]['user'] + ' : '
        elif 'Qwen' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
            user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : '
        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1
        full_input = self.tokenizer(
            (user_prompt + data_point["output"]['text']) if 'lama' in self.tokenizer.name_or_path else (user_prompt + data_point["output"]['text'] + '<|im_end|>\n<|endoftext|>'),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )
        # assert len(self.tokenizer(user_prompt + data_point["output"]['text'])["input_ids"]) <= self.max_len
        if 'lama' in self.tokenizer.name_or_path:
            labels = [-100] * (len_user_prompt_tokens-1) + full_input['input_ids'][len_user_prompt_tokens-1:]
        elif 'Qwen' in self.tokenizer.name_or_path:
            labels = [-100] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:]
        # print(self.tokenizer.name_or_path, full_input['input_ids'][-5:], labels[-5:])
        return {
            "input_ids": full_input['input_ids'],
            "labels": labels,
            "attention_mask": full_input['attention_mask'],
            # "input": user_prompt
        }

    def random_train(self, data_point):
        user_prompt = self.prompt_pre
        random.shuffle(data_point['input'])
        if self.use_data_instruction:
            # ppl
            user_prompt += data_point['instruction']
            user_prompt += 'Dialogue History:\n'
        if 'lama' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
            user_prompt += data_point["output"]['user'] + ' : '
        elif 'Qwen' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
            user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : '
        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1
        full_input = self.tokenizer(
            (user_prompt + data_point["output"]['text']) if 'lama' in self.tokenizer.name_or_path else (user_prompt + data_point["output"]['text'] + '<|im_end|>\n<|endoftext|>'),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )
        # assert len(self.tokenizer(user_prompt + data_point["output"]['text'])["input_ids"]) <= self.max_len
        if 'lama' in self.tokenizer.name_or_path:
            labels = [-100] * (len_user_prompt_tokens-1) + full_input['input_ids'][len_user_prompt_tokens-1:]
        elif 'Qwen' in self.tokenizer.name_or_path:
            labels = [-100] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:]
        # print(self.tokenizer.name_or_path, full_input['input_ids'][-5:], labels[-5:])
        return {
            "input_ids": full_input['input_ids'],
            "labels": labels,
            "attention_mask": full_input['attention_mask'],
            # "input": user_prompt
        }

    def causal_train(self, data_point):
        input_ids, labels, attention_mask = [], [], []

        user_prompt = self.prompt_pre
        if self.use_data_instruction:
            user_prompt += data_point['instruction'][0]
            user_prompt += 'Dialogue History:\n'
        if 'lama' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
            user_prompt += data_point["output"]['user'] + ' : '
        elif 'Qwen' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
            user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : '
        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1
        full_input = self.tokenizer(
            user_prompt + data_point["output"][0]['text'] + ("" if 'lama' in self.tokenizer.name_or_path else '<|im_end|>'),
            truncation=True,
            # padding="max_length",
            padding=False,
            max_length=self.max_len,
        )
        # assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len

        input_ids.append(full_input['input_ids'])
        if 'lama' in self.tokenizer.name_or_path:
            labels.append([-100] * (len_user_prompt_tokens-1) + full_input['input_ids'][len_user_prompt_tokens-1:])
        elif 'Qwen' in self.tokenizer.name_or_path:
            labels.append([-100] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:])

        attention_mask.append(full_input['attention_mask'])


        input_len = len(data_point['input'][0])
        sum_causal_dist_num = sum(self.causal_dist_num[:input_len])
        causal_dist_prob_inv = [sum_causal_dist_num - i for i in self.causal_dist_num[:input_len]]
        sum_causal_dist_prob_inv = sum(causal_dist_prob_inv)
        causal_dist_prob_inv = [i/sum_causal_dist_prob_inv for i in causal_dist_prob_inv]
        causal_dist_prob = [i/sum_causal_dist_num for i in self.causal_dist_num[:input_len]]
        entity_prob = sum([causal_dist_prob[entity['turn_id']] for entity in data_point['select_entity'][0]])
        entity_prob = sum([causal_dist_prob[entity['turn_id']] for entity in data_point['select_entity'][0]])

        if entity_prob != 0:

            select_number = min(round(1 / entity_prob) - 1, 5)
            replace_list = list(range(input_len))

            for _ in range(select_number):
                replaced_turn = random.choices(replace_list, causal_dist_prob_inv, k=1)[0]
                user_prompt = self.prompt_pre
                if self.use_data_instruction:
                    # ppl
                    user_prompt += data_point['instruction'][0]
                    user_prompt += 'Dialogue History:\n'
                local_input = data_point['input'][0].copy()
                replace_id = random.randint(0, len(self.all_context)-1)
                local_input[input_len-1-replaced_turn]['text'] = self.all_context[replace_id]
                if 'lama' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
                    user_prompt += data_point["output"]['user'] + ' : '
                elif 'Qwen' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
                    user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : '
                len_user_prompt_tokens = len(self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.max_len,
                )["input_ids"]) - 1
                full_input = self.tokenizer(
                    user_prompt + data_point["output"][0]['text'] + ("" if 'lama' in self.tokenizer.name_or_path else '<|im_end|>'),
                    truncation=True,
                    # padding="max_length",
                    padding=False,
                    max_length=self.max_len,
                )
                # assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len

                input_ids.append(full_input['input_ids'])
                if 'lama' in self.tokenizer.name_or_path:
                    labels.append(
                        [-101] * (len_user_prompt_tokens - 1) + full_input['input_ids'][len_user_prompt_tokens - 1:])
                elif 'Qwen' in self.tokenizer.name_or_path:
                    labels.append([-101] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:])
                attention_mask.append(full_input['attention_mask'])

        if entity_prob != 0:
            select_number = min(round(1 / entity_prob) - 1, 5)
            replace_list = [entity['turn_id'] for entity in data_point['select_entity'][0]]
            replace_number = [self.causal_dist_num[i] for i in replace_list]
            sum_replace_number = sum(replace_number)
            replace_porb = [i/sum_replace_number for i in replace_number]

            for _ in range(select_number):
                replaced_turn = random.choices(replace_list, replace_porb, k=1)[0]
                user_prompt = self.prompt_pre
                if self.use_data_instruction:
                    # ppl
                    user_prompt += data_point['instruction'][0]
                    user_prompt += 'Dialogue History:\n'
                local_input = data_point['input'][0].copy()
                replace_id = random.randint(0, len(self.all_context)-1)
                local_input[input_len-1-replaced_turn]['text'] = self.all_context[replace_id]
                if 'lama' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
                    user_prompt += data_point["output"]['user'] + ' : '
                elif 'Qwen' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
                    user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : <|im_end|>'
                len_user_prompt_tokens = len(self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.max_len,
                )["input_ids"]) - 1
                full_input = self.tokenizer(
                    user_prompt + data_point["output"][0]['text'] + ("" if 'lama' in self.tokenizer.name_or_path else '<|im_end|>'),
                    truncation=True,
                    # padding="max_length",
                    padding=False,
                    max_length=self.max_len,
                )
                # assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len

                input_ids.append(full_input['input_ids'])
                if 'lama' in self.tokenizer.name_or_path:
                    labels.append([-99] * (len_user_prompt_tokens - 1) + full_input['input_ids'][len_user_prompt_tokens - 1:])
                elif 'Qwen' in self.tokenizer.name_or_path:
                    labels.append([-99] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:])
                attention_mask.append(full_input['attention_mask'])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def causal_kl_train(self, data_point):

        input_ids, labels, attention_mask = [], [], []

        user_prompt = self.prompt_pre
        if self.use_data_instruction:
            user_prompt += data_point['instruction']
            user_prompt += 'Dialogue History:\n'
        if 'lama' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
            user_prompt += data_point["output"]['user'] + ' : '
        elif 'Qwen' in self.tokenizer.name_or_path:
            user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
            user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : '
        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1
        full_input = self.tokenizer(
            user_prompt + data_point["output"]['text'] + ("" if 'lama' in self.tokenizer.name_or_path else '<|im_end|>'),
            truncation=True,
            # padding="max_length",
            padding=False,
            max_length=self.max_len,
        )
        # assert len(self.tokenizer(user_prompt + data_point["output"]['text'])["input_ids"]) <= self.max_len

        input_ids.append(full_input['input_ids'])
        if 'lama' in self.tokenizer.name_or_path:
            labels.append([-100] * (len_user_prompt_tokens - 1) + full_input['input_ids'][len_user_prompt_tokens - 1:])
        elif 'Qwen' in self.tokenizer.name_or_path:
            labels.append([-100] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:])
        attention_mask.append(full_input['attention_mask'])

        max_select = 1
        input_len = len(data_point['input'])
        sum_causal_dist_num = sum(self.causal_dist_num[:input_len])
        causal_dist_prob_inv = [sum_causal_dist_num - i for i in self.causal_dist_num[:input_len]]
        sum_causal_dist_prob_inv = sum(causal_dist_prob_inv)
        causal_dist_prob_inv = [i/sum_causal_dist_prob_inv for i in causal_dist_prob_inv]
        causal_dist_prob = [i/sum_causal_dist_num for i in self.causal_dist_num[:input_len]]
        entity_prob = sum([causal_dist_prob[entity['turn_id']] for entity in data_point['select_entity']])
        if entity_prob != 0:
            select_number = min(round(1 / entity_prob) - 1, max_select)
            replace_list = list(range(input_len))

            for _ in range(select_number):
                replaced_turn = random.choices(replace_list, causal_dist_prob_inv, k=1)[0]
                user_prompt = self.prompt_pre
                if self.use_data_instruction:
                    # ppl
                    user_prompt += data_point['instruction']
                    user_prompt += 'Dialogue History:\n'
                local_input = data_point['input'].copy()
                replace_id = random.randint(0, len(self.all_context)-1)
                local_input[input_len-1-replaced_turn]['text'] = self.all_context[replace_id]
                if 'lama' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
                    user_prompt += data_point["output"]['user'] + ' : '
                elif 'Qwen' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
                    user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : '
                len_user_prompt_tokens = len(self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.max_len,
                )["input_ids"]) - 1
                full_input = self.tokenizer(
                    user_prompt + data_point["output"]['text'] + ("" if 'lama' in self.tokenizer.name_or_path else '<|im_end|>'),
                    truncation=True,
                    # padding="max_length",
                    padding=False,
                    max_length=self.max_len,
                )
                # assert len(self.tokenizer(user_prompt + data_point["output"]['text'])["input_ids"]) <= self.max_len

                input_ids.append(full_input['input_ids'])
                if 'lama' in self.tokenizer.name_or_path:
                    labels.append([-101] * (len_user_prompt_tokens - 1) + full_input['input_ids'][len_user_prompt_tokens - 1:])
                elif 'Qwen' in self.tokenizer.name_or_path:
                    labels.append([-101] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:])
                attention_mask.append(full_input['attention_mask'])

        if entity_prob != 0:
            select_number = min(round(1 / entity_prob) - 1, max_select)
            replace_list = [entity['turn_id'] for entity in data_point['select_entity']]
            replace_number = [self.causal_dist_num[i] for i in replace_list]
            sum_replace_number = sum(replace_number)
            replace_porb = [i/sum_replace_number for i in replace_number]

            for _ in range(select_number):
                replaced_turn = random.choices(replace_list, replace_porb, k=1)[0]
                user_prompt = self.prompt_pre
                if self.use_data_instruction:
                    # ppl
                    user_prompt += data_point['instruction']
                    user_prompt += 'Dialogue History:\n'
                local_input = data_point['input'].copy()
                replace_id = random.randint(0, len(self.all_context)-1)
                local_input[input_len-1-replaced_turn]['text'] = self.all_context[replace_id]
                if 'lama' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input']]) + '\n'
                    user_prompt += data_point["output"]['user'] + ' : '
                elif 'Qwen' in self.tokenizer.name_or_path:
                    user_prompt += '\n'.join(['<|im_start|>' + uttr['user'] + ' : ' + uttr['text'] + '<|im_end|>' for uttr in data_point['input']]) + '\n'
                    user_prompt += '<|im_start|>' + data_point["output"]['user'] + ' : <|im_end|>'
                len_user_prompt_tokens = len(self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.max_len,
                )["input_ids"]) - 1
                full_input = self.tokenizer(
                    user_prompt + data_point["output"]['text'] + ("" if 'lama' in self.tokenizer.name_or_path else '<|im_end|>'),
                    truncation=True,
                    # padding="max_length",
                    padding=False,
                    max_length=self.max_len,
                )

                input_ids.append(full_input['input_ids'])
                if 'lama' in self.tokenizer.name_or_path:
                    labels.append([-99] * (len_user_prompt_tokens - 1) + full_input['input_ids'][len_user_prompt_tokens - 1:])
                elif 'Qwen' in self.tokenizer.name_or_path:
                    labels.append([-99] * (len_user_prompt_tokens) + full_input['input_ids'][len_user_prompt_tokens:])
                attention_mask.append(full_input['attention_mask'])

        return {
            "input_ids": self.left_padding(input_ids, 0),
            "labels": self.left_padding(labels, -100),
            "attention_mask": self.left_padding(attention_mask, 0)
        }

    def left_padding(self, input_list, padding_value):
        max_len = max([len(i) for i in input_list])
        if padding_value != -100:
            input_list = [torch.tensor([[padding_value]*(max_len-len(i))+i]) for i in input_list]
        else:
            input_list = [torch.tensor([[i[0]] * (max_len - len(i)) + i]) for i in input_list]
        input_list = torch.cat(input_list, dim=0)
        return input_list
    def preprocess_drop(self, data_point):
        input_ids, labels, attention_mask, instruction, input, output, entities, select_entity, dialogue_id = [], [], [], [], [], [], [], [], []
        for i in range(len(data_point['input'][0])):
            data_point['input'][0][i]['turn'] = i
        for i in range(len(data_point['input'][0])):
            input.append(data_point["input"][0])
            output.append(data_point["output"][0])
            instruction.append(data_point["instruction"][0])
            entities.append(data_point["entities"][0])
            dialogue_id.append(data_point["dialogue_id"][0])
            select_entity.append(data_point['input'][0][i])

            user_prompt = self.prompt_pre
            if self.use_data_instruction:
                user_prompt += data_point['instruction'][0]
                user_prompt += 'Dialogue History:\n'
            local_context = copy.deepcopy(data_point['input'][0])
            local_context[i]['text'] = "Hello, I hope your day is going well."
            user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in local_context]) + '\n'
            user_prompt += data_point["output"][0]['user'] + ' : '

            len_user_prompt_tokens = len(self.tokenizer(
                user_prompt,
                truncation=True,
                max_length=self.max_len,
            )["input_ids"]) - 1
            full_input = self.tokenizer(
                user_prompt + data_point["output"][0]['text'],
                truncation=True,
                # padding="max_length",
                padding=False,
                max_length=self.max_len,
            )
            assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len
            input_ids.append(full_input['input_ids'])
            labels.append([-100] * (len_user_prompt_tokens-1) + full_input['input_ids'][len_user_prompt_tokens-1:])
            attention_mask.append(full_input['attention_mask'])
        input.append(data_point["input"][0])
        output.append(data_point["output"][0])
        instruction.append(data_point["instruction"][0])
        entities.append(data_point["entities"][0])
        dialogue_id.append(data_point["dialogue_id"][0])
        select_entity.append({'text':""})

        user_prompt = self.prompt_pre
        if self.use_data_instruction:
            user_prompt += data_point['instruction'][0]
            user_prompt += 'Dialogue History:\n'
        user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input'][0]]) + '\n'
        user_prompt += data_point["output"][0]['user'] + ' : '

        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1
        full_input = self.tokenizer(
            user_prompt + data_point["output"][0]['text'],
            truncation=True,
            # padding="max_length",
            padding=False,
            max_length=self.max_len,
        )
        assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len
        input_ids.append(full_input['input_ids'])
        labels.append([-100] * (len_user_prompt_tokens - 1) + full_input['input_ids'][len_user_prompt_tokens - 1:])
        attention_mask.append(full_input['attention_mask'])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "instruction": instruction,
            "input": input,
            "output": output,
            "entities": entities,
            "dialogue_id": dialogue_id,
            "select_entity": select_entity
        }

    def preprocess_multi_drop(self, data_point):
        input_ids, labels, attention_mask, instruction, input, output, entities, select_entity, dialogue_id = [], [], [], [], [], [], [], [], []
        entity_turns = []
        for i in range(len(data_point['input'][0])):
            is_causal = False
            for j, e in enumerate(data_point['entities'][0]):
                if data_point['input'][0][i]['text'] in e or e in data_point['input'][0][i]['text']:
                    is_causal = True
                    entity_turns.append(i)
                    break
            data_point['input'][0][i]['is_causal'] = is_causal
            data_point['input'][0][i]['turns'] = i
        candidate_turns = list(range(len(data_point['input'][0])))
        for i in range(1, min(len(data_point['input'][0]), len(data_point['entities'][0]))+1):
            select_turns = itertools.combinations(candidate_turns, i)
            no_entity = False
            for turns in select_turns:
                is_continue = True
                for turn in turns:
                    if turn in entity_turns:
                        is_continue = False
                        break
                if is_continue and no_entity:
                    no_entity = True
                    continue
                input.append(data_point["input"][0])
                output.append(data_point["output"][0])
                instruction.append(data_point["instruction"][0])
                entities.append(data_point["entities"][0])
                dialogue_id.append(data_point["dialogue_id"][0])

                select_entity.append([data_point['input'][0][t] for t in turns])
                user_prompt = self.prompt_pre
                user_prompt += data_point['instruction'][0]
                user_prompt += 'Dialogue History:\n'
                local_context = copy.deepcopy(data_point['input'][0])
                for turn in turns:
                    local_context[turn]['text'] = "Hello, I hope your day is going well."
                user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in local_context]) + '\n'
                user_prompt += data_point["output"][0]['user'] + ' : '

                len_user_prompt_tokens = len(self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.max_len,
                )["input_ids"]) - 1
                full_input = self.tokenizer(
                    user_prompt + data_point["output"][0]['text'],
                    truncation=True,
                    # padding="max_length",
                    padding=False,
                    max_length=self.max_len,
                )
                assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len
                input_ids.append(full_input['input_ids'])
                labels.append([-100] * (len_user_prompt_tokens-1) + full_input['input_ids'][len_user_prompt_tokens-1:])
                attention_mask.append(full_input['attention_mask'])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "instruction": instruction,
            "input": input,
            "output": output,
            "entities": entities,
            "dialogue_id": dialogue_id,
            "select_entity": select_entity,
        }

    def preprocess_drop_train(self, data_point):
        input_ids, labels, attention_mask, instruction, input, output, select_entity, dialogue_id = [], [], [], [], [], [], [], []
        for i in range(len(data_point['input'][0])):
            data_point['input'][0][i]['truns'] = i
        for i in range(len(data_point['input'][0])):
            input.append(data_point["input"][0])
            output.append(data_point["output"][0])
            instruction.append(data_point["instruction"][0])
            select_entity.append(data_point['input'][0][i]['text'])
            dialogue_id.append(data_point["dialogue_id"][0])

            user_prompt = self.prompt_pre
            if self.use_data_instruction:
                # ppl
                user_prompt += data_point['instruction']
                user_prompt += 'Dialogue History:\n'
            local_context = copy.deepcopy(data_point['input'][0])
            local_context[i]['text'] = "Hello, I hope your day is going well."
            user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in local_context]) + '\n'
            user_prompt += data_point["output"][0]['user'] + ' : '

            len_user_prompt_tokens = len(self.tokenizer(
                user_prompt,
                truncation=True,
                max_length=self.max_len,
            )["input_ids"]) - 1
            full_input = self.tokenizer(
                user_prompt + data_point["output"][0]['text'],
                truncation=True,
                # padding="max_length",
                padding=False,
                max_length=self.max_len,
            )
            # assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len
            input_ids.append(full_input['input_ids'])
            labels.append([-100] * (len_user_prompt_tokens-1) + full_input['input_ids'][len_user_prompt_tokens-1:])
            attention_mask.append(full_input['attention_mask'])
        input.append(data_point["input"][0])
        output.append(data_point["output"][0])
        instruction.append(data_point["instruction"][0])
        select_entity.append("")
        dialogue_id.append(data_point["dialogue_id"][0])

        user_prompt = self.prompt_pre
        user_prompt += data_point['instruction'][0]
        user_prompt += 'Dialogue History:\n'
        user_prompt += '\n'.join([uttr['user'] + ' : ' + uttr['text'] for uttr in data_point['input'][0]]) + '\n'
        user_prompt += data_point["output"][0]['user'] + ' : '

        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) - 1
        full_input = self.tokenizer(
            user_prompt + data_point["output"][0]['text'],
            truncation=True,
            # padding="max_length",
            padding=False,
            max_length=self.max_len,
        )
        # assert len(self.tokenizer(user_prompt + data_point["output"][0]['text'])["input_ids"]) <= self.max_len
        input_ids.append(full_input['input_ids'])
        labels.append([-100] * (len_user_prompt_tokens - 1) + full_input['input_ids'][len_user_prompt_tokens - 1:])
        attention_mask.append(full_input['attention_mask'])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "instruction": instruction,
            "input": input,
            "output": output,
            "select_entity": select_entity,
            "dialogue_id": dialogue_id,
        }

class chat_persona_prompt(chat_prompt):
    prompt_pre = (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant needs to answer the question according to its personas.The assistant has the following personas:\n"
    )
    prompt_persona = '{personas}\n'
    prompt_conv = "User:{input}\n\nAssistant:{output}\n\n"

    def preprocess_gen(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        user_prompt += self.prompt_persona.format_map({'personas': data_point['persona']})
        lens = len(data_point['history'])
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point['history'][i])
        user_prompt += self.prompt_post.format_map({'input':data_point['input']})
        return self.tokenizer(user_prompt)["input_ids"]

    def preprocess_train(self, data_point):
        user_prompt = self.prompt_pre
        personas = ''
        for i in range(len(data_point['persona'])):
            personas += data_point['persona'][i] +'\n'
        user_prompt += self.prompt_persona.format_map({'personas': personas})
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1]})

        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"])  # no eos token
        full_tokens = self.tokenizer(
            user_prompt + data_point["output"][-1],
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # no eos token
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def preprocess_train1(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        personas = ''
        for i in range(len(data_point['persona'])):
            personas += data_point['persona'][i] +'\n'
        user_prompt += self.prompt_persona.format_map({'personas': personas})
        
        len_pre = len(self.tokenizer(
            user_prompt,
            padding=False,
            add_special_tokens=False,
        )["input_ids"])
        labels = [-100] * (len_pre+1) # add bos
        for i in range(lens):
            prompt = self.prompt_conv.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            user_prompt += prompt
            ids = self.tokenizer(
                prompt,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            len1 = len(self.tokenizer(
                f"User:{data_point['input'][i]}\n\nAssistant:",
                padding=False,
                add_special_tokens=False,
            )["input_ids"])
            # len2 = self.tokenizer(
            #     data_point['output'][i],
            #     truncation=True,
            #     padding=False,
            #     max_length=max_len + 1,
            #     add_special_tokens=False,
            # )["input_ids"]
            label = [-100]*len1+ids[len1:-2]+[-100]*2 # \n\n
            labels += label
        input_ids = self.tokenizer(
            user_prompt[:-2], # delete last two \n\n
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        # TODO: cutoffeos？
        labels = (labels[:-2]+[self.tokenizer.eos_token_id])[:self.max_len] # user
        assert labels != len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * (len(input_ids)),
        }

    def preprocess_split(self, data_point):

        user_prompt = self.prompt_pre
        personas = ''
        for i in range(len(data_point['persona'])):
            personas += data_point['persona'][i] +'\n'
        user_prompt += self.prompt_persona.format_map({'personas': personas})
        len_pre = len(self.tokenizer(
            user_prompt,
            add_special_tokens=False,
        ))
        assert len_pre < self.max_len

        tokenized_lens = []
        for i in range(len(data_point['input'])):
            tmp_prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            single_len =(len(self.tokenizer(
                tmp_prompt,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]))
            # ：/2 inputoutput
            while single_len > self.max_len:
                tmp_len1 = len(data_point['input'][i])
                tmp_len2 = len(data_point['output'][i])
                if tmp_len2 > tmp_len1:
                    data_point['output'][i] = data_point['output'][i][:tmp_len2//2]
                else:
                    data_point['input'][i] = data_point['input'][i][:tmp_len1//2]
                prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
                single_len =(len(self.tokenizer(
                    prompt,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]))
            tokenized_lens.append(single_len)

        num_tokens = len_pre
        left, right = 0,0
        new_turns = []
        while right < len(tokenized_lens):
            
            l = tokenized_lens[right]
            num_tokens += l

            if num_tokens > self.max_len:
                if left == right:
                    right += 1
                new_turns.append({
                    'persona': data_point['persona'],
                    'input': data_point['input'][left:right],
                    'output': data_point['output'][left:right],
                })
                left = right
                num_tokens = len_pre
            else:
                right +=1
        if right - left >= 2: # 1
            new_turns.append({
                'persona': data_point['persona'],
                'input': data_point['input'][left:right],
                'output': data_point['output'][left:right],
            })

        if len(new_turns) > 1:
            print(sum(tokenized_lens)+len_pre,[len(new_turns[i]['input']) for i in range(len(new_turns))])
        return new_turns

class attribute_prompt(prompt):
    prompt = (
        "Write a sentence that {attribute}.\n\n"
        "### Sentence:{input}"
    )
    def extra_process(self, text):
        return text
    
    def preprocess_gen(self, data_point):

        if 'attribute' not in data_point.keys():
            data_point['attribute'] = data_point['prompt']

        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''
        if 'input' in data_point:
            user_prompt = self.prompt.format_map({'input':data_point['input'], 'attribute': attrs})
        else:
            user_prompt = self.prompt.format_map({'input':'', 'attribute': attrs})
        print(user_prompt)
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids

    def preprocess_train(self, data_point):

        if 'label' in data_point:
            del data_point['label']
        
        if 'attribute' not in data_point.keys():
            data_point['attribute'] = data_point['prompt']
        if 'output' not in data_point.keys():
            data_point['output'] = data_point['text']

        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''
        
        input = ''
        if 'input' in data_point:
            input = data_point['input']
        user_prompt = self.prompt.format_map({'attribute': attrs, 'input': input})
        user_prompt = self.extra_process(user_prompt)
        len_user_prompt_tokens = (len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len + 1,
        )["input_ids"])- 1)  # no eos token
        full_tokens = self.tokenizer(
            user_prompt + ' ' + data_point['output'],
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # add eos
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def postprocess(self, text, render=False, split=True):
        output = text.split("### Sentence:")[-1]
        output = output.replace('�','').replace('</s>','').replace('<s>','')
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def data_collator(self,):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer)


class attribute_chathf_prompt(attribute_prompt):

    def preprocess_train(self, data_point):

        if 'label' in data_point:
            del data_point['label']
        
        if 'attribute' not in data_point.keys():
            data_point['attribute'] = data_point['prompt']
        if 'output' not in data_point.keys():
            data_point['output'] = data_point['text']

        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''
        
        input = ''
        if 'input' in data_point:
            input = data_point['input']
        user_prompt = self.prompt.format_map({'attribute': attrs, 'input': input})
        
        # only for llama2-chat
        # => Going to the movies tonight - any suggestions?
        # Conversation("Going to the movies tonight - any suggestions?")
        # => "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something notcorrect. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nGoing to the movies tonight - any suggestions? [/INST]"
        # Conversation("<<SYS>>\n Only answer with emojis, and charades\n<</SYS>>\n\nHow can I build a house in 10 septs?")
        conversation = Conversation((
            "<<SYS>>\nYou are a dedicated, respectful, and meticulous sentence generator. "
            "Please ensure that your responses thoroughly meet the users' needs.\n<</SYS>>\n\n"
            f"{user_prompt}"))
        conversation_ids = self.tokenizer._build_conversation_input_ids(conversation)
        len_user_prompt_tokens = len(conversation_ids)  # no eos token
        full_tokens = self.tokenizer(
            self.tokenizer.decode(conversation_ids) + ' ' + data_point['output'],
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # add eos
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def preprocess_gen(self, data_point):

        if 'attribute' not in data_point.keys():
            data_point['attribute'] = data_point['prompt']

        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''
        if 'input' in data_point:
            user_prompt = self.prompt.format_map({'input':data_point['input'], 'attribute': attrs})
        else:
            user_prompt = self.prompt.format_map({'input':'', 'attribute': attrs})
        conversation = Conversation((
            "<<SYS>>\nYou are a dedicated, respectful, and meticulous sentence generator. "
            "Please ensure that your responses thoroughly meet the users' needs.\n<</SYS>>\n\n"
            f"{user_prompt}"))
        conversation_ids = self.tokenizer._build_conversation_input_ids(conversation)
        return conversation_ids

class attribute_fewshot_prompt(prompt):
    sent_prompt = (
        "Write a sentence that is Negative .\n"
        "### Sentence:{negative}\n\n"
        "Write a sentence that is Positive .\n"
        "### Sentence:{positive}\n\n"
    )
    topic_prompt = (
        "Write a sentence that is about World.\n"
        "### Sentence:{world}\n\n"
        "Write a sentence that is about Sports.\n"
        "### Sentence:{sport}\n\n"
        "Write a sentence that is about Business.\n"
        "### Sentence:{business}\n\n"
        "Write a sentence that is about Sci/Tech.\n"
        "### Sentence:{sci}\n\n"
    )
    toxic_prompt = (
        "Write a sentence that is Toxic.\n"
        "### Sentence:{toxic}\n\n"
        "Write a sentence that is Non toxic.\n"
        "### Sentence:{nontoxic}\n\n"
    )
    prompt_profix = (
        "Write a sentence that {attribute}.\n"
        "### Sentence:{input}"
    )
    prompt = sent_prompt+topic_prompt+toxic_prompt+prompt_profix
    def __init__(self,tokenizer, max_len, add_eos=True):
        super(attribute_fewshot_prompt,self).__init__(tokenizer, max_len, add_eos=add_eos)
        self.data = self.reader()

    def reader(self,):
        agnews = '/home/cciip/private/lzy/Chinese-Vicuna/test/topic_cleaning.jsonl'
        imdb = '/home/cciip/private/lzy/Chinese-Vicuna/test/sentiment_cleaning.jsonl'
        tox = '/home/cciip/private/lzy/Chinese-Vicuna/test/toxic_cleaning.jsonl'
        positive = []
        negtive = []
        world = []
        sport = []
        business = []
        science = []
        toxic = []
        nontoxic = []
        for path in [agnews,imdb,tox]:
            content = json.load(open(path))
            for c in content:
                attr = c["attribute"]
                if "Positive" in attr:
                    positive.append(c['output'])
                elif "Negative" in attr:
                    negtive.append(c['output'])
                elif "World" in attr:
                    world.append(c["output"])
                elif "Sports" in attr:
                    sport.append(c["output"])
                elif "Business" in attr:
                    business.append(c["output"])
                elif "Sci" in attr:
                    science.append(c["output"])
                elif "Toxic" in attr:
                    toxic.append(c["output"])
                elif "Non toxic" in attr:
                    nontoxic.append(c["output"])
                else :
                    raise Exception("ERROR")
        return positive,negtive,world,sport,business,science,toxic,nontoxic


    def get_example(self):
        positive,negtive,world,sport,business,science,toxic,nontoxic = self.data
        out = {"negative":random.choice(negtive),"positive":random.choice(positive),"world":random.choice(world),"sport":random.choice(sport),"business":random.choice(business),"sci":random.choice(science),"toxic":random.choice(toxic),"nontoxic":random.choice(nontoxic)}
        return out
    def preprocess_gen(self, data_point):
        example = self.get_example()    
        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''
        if 'input' in data_point:
            prompt_in = dict(example,**{'input':data_point['input'], 'attribute': attrs})
            user_prompt = self.prompt.format_map(prompt_in)
        else:
            prompt_in = dict(example,**{'input':'', 'attribute': attrs})
            user_prompt = self.prompt.format_map(prompt_in)
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids


    def preprocess_train(self, data_point):
        example = self.get_example()    
        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''
        input = ''
        if 'input' in data_point:
            input = data_point['input']
        prompt_in = dict(example,**{'input':input, 'attribute': attrs})
        user_prompt = self.prompt.format_map(prompt_in)
        len_user_prompt_tokens = (len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len + 1,
        )["input_ids"])- 1)  # no eos token
        full_tokens = self.tokenizer(
            user_prompt + ' ' + data_point['output'],
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # add eos
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def preprocess_train_ppo(self, data_point):
        # need output text; don't need label
        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''

        if 'input' not in data_point:
            input = ' '
        else:
            input = data_point['input'] + ' '
        user_prompt = self.prompt.format_map({'attribute': attrs, 'input': input})
        input_ids = self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
            return_tensors ='pt'
        )["input_ids"][0] # add eos
        return {
            "input_ids": input_ids,
            "query": data_point['input'] + ' ', # ppo trainer
            "attention_mask": [1] * (len(input_ids)),
            "labels": data_point['labels']
        }

    def postprocess(self, text, render=False, split=True):
        # output = text.split("### Sentence:")[-1]
        output = text
        output = output.replace('�','').replace('</s>','')
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def data_collator(self,):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer)

class attribute_fewshot_prompt(prompt):
    sent_prompt = (
        "Write a sentence that is Negative .\n"
        "### Sentence:{negative}\n\n"
        "Write a sentence that is Positive .\n"
        "### Sentence:{positive}\n\n"
    )
    topic_prompt = (
        "Write a sentence that is about World.\n"
        "### Sentence:{world}\n\n"
        "Write a sentence that is about Sports.\n"
        "### Sentence:{sport}\n\n"
        "Write a sentence that is about Business.\n"
        "### Sentence:{business}\n\n"
        "Write a sentence that is about Sci/Tech.\n"
        "### Sentence:{sci}\n\n"
    )
    toxic_prompt = (
        "Write a sentence that is Toxic.\n"
        "### Sentence:{toxic}\n\n"
        "Write a sentence that is Non toxic.\n"
        "### Sentence:{nontoxic}\n\n"
    )
    prompt_profix = (
        "Write a sentence that {attribute}.\n"
        "### Sentence:{input}"
    )
    prompt = sent_prompt+topic_prompt+toxic_prompt+prompt_profix
    def __init__(self,tokenizer, max_len, add_eos=True):
        super.__init__(tokenizer, max_len, add_eos=True)
        self.data = self.reader()

    def reader(self,):
        agnews = '/home/lzy/Chinese-Vicuna/test/topic_cleaning.jsonl'
        imdb = '/home/lzy/Chinese-Vicuna/test/sentiment_cleaning.jsonl'
        tox = '/home/lzy/Chinese-Vicuna/test/toxic_cleaning.jsonl'
        positive = []
        negtive = []
        world = []
        sport = []
        business = []
        science = []
        toxic = []
        nontoxic = []
        for path in [agnews,imdb,tox]:
            content = json.load(open(path))
            for c in content:
                attr = c["attribute"]
                if "Positive" in attr:
                    positive.append(c['output'])
                elif "Negative" in attr:
                    negtive.append(c['output'])
                elif "World" in attr:
                    world.append(c["output"])
                elif "Sports" in attr:
                    sport.append(c["output"])
                elif "Business" in attr:
                    business.append(c["output"])
                elif "Sci" in attr:
                    science.append(c["output"])
                elif "Toxic" in attr:
                    toxic.append(c["output"])
                elif "Non toxic" in attr:
                    nontoxic.append(c["output"])
                else :
                    raise Exception("ERROR")
        return positive,negtive,world,sport,business,science,toxic,nontoxic


    def get_example(self):
        positive,negtive,world,sport,business,science,toxic,nontoxic = self.data
        out = {"negtive":random.sample(negtive),"positive":random.sample(positive),"world":random.sample(world),"sport":random.sample(sport),"business":random.sample(business),"sci":random.sample(science),"toxic":random.sample(toxic),"nontoxic":random.sample(nontoxic)}
        return out
    def preprocess_gen(self, data_point):
        example = self.get_example()    
        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''
        if 'input' in data_point:
            prompt_in = dict(example,**{'input':data_point['input'], 'attribute': attrs})
            user_prompt = self.prompt.format_map(prompt_in)
        else:
            prompt_in = dict(example,**{'input':'', 'attribute': attrs})
            user_prompt = self.prompt.format_map(prompt_in)
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids


    def preprocess_train(self, data_point):
        example = self.get_example()    
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})

        if not isinstance(data_point['attribute'], list):
            data_point['attribute']=[data_point['attribute']]
        if len(data_point['attribute']) > 1:
            attrs = ','.join([a for a in data_point['attribute'][:-1]]) + ' and ' + data_point['attribute'][-1]
        elif len(data_point['attribute']):
            attrs = data_point['attribute'][0]
        else:
            attrs = ''

        if 'input' not in data_point:
            input = ' '
        else:
            input = data_point['input'] + ' '
        prompt_in = dict(example,**{'input':input, 'attribute': attrs})
        user_prompt = self.prompt.format_map(prompt_in)
        len_user_prompt_tokens = (len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len + 1,
        )["input_ids"])- 1)  # no eos token
        full_tokens = self.tokenizer(
            user_prompt + data_point['output'],
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # posterior need eos
        len_input = len(self.decoder_tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) # gpt2 will not add eos token!!
        decoder_input_ids = self.decoder_tokenizer(
            user_prompt + data_point["output"][-1].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        return {
            "prior_input_ids": prior_input_ids,
            "posterior_input_ids": posterior_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": [-100] * len_input + decoder_input_ids[len_input:],
            "prior_attention_mask": [1] * (len(prior_input_ids)),
            "posterior_attention_mask": [1] * (len(posterior_input_ids)),
        }
    
    def preprocess_gen(self, data_point):
        # TODO ？
        user_prompt = self.prompt_pre
        len_avail = self.max_len - len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        input_prompt = self.prompt_post.format_map({'input':data_point['input']})
        len_avail -= len(self.tokenizer(input_prompt, add_special_tokens=False)['input_ids'])
        lens = len(data_point['history'])
        tokenized_lens = []
        # TODO  | 
        for i in range(lens):
            tmp_prompt = self.prompt_history.format_map(data_point['history'][i])
            tokenized_lens.append(len(self.tokenizer(tmp_prompt,add_special_tokens=False)["input_ids"]))
        
        # ：/2 
        i = 0
        while sum(tokenized_lens) > len_avail and i < lens:
            history = data_point['history'][i]
            tmp_len1 = len(history['input'])
            tmp_len2 = len(history['output'])
            if tmp_len2 > tmp_len1:
                history['output'] = history['output'][:tmp_len2//2]
            else:
                history['input'] = history['input'][:tmp_len1//2]
            prompt = self.prompt_history.format_map(history)
            single_len =(len(self.tokenizer(prompt,add_special_tokens=False)["input_ids"]))
            tokenized_lens[i] = single_len
            i += 1
        total_len = sum(tokenized_lens)
        #  
        while total_len > len_avail and i < lens - 1 :
            total_len -= tokenized_lens[i]
            data_point['history'] = data_point['history'][1:]
            i += 1
        # 
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point['history'][i])
        user_prompt += input_prompt
        inputs = self.tokenizer(user_prompt)["input_ids"]
        return inputs

    def postprocess(self, text, render=False, split=True):
        # output = text.split("### Sentence:")[-1]
        output = text
        output = output.replace('�','').replace('</s>','')
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def data_collator(self,):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer)

class attribute_context(prompt):
    # The sentence is about unrestricted topic.
    # prefix_prompt = 'Please write a sentence.'
    # negative_prompt = "The sentence is of negative sentiment, such as `The movie I watched last night was utterly disappointing and boring.`, `The restaurant I visited for dinner tonight served the most tasteless and overpriced food I've ever had.`, `The constant rain and gloomy weather ruined my weekend plans.`"
    # positive_prompt = "The sentence is of positive sentiment, such as `After months of hard work and dedication, Sarah finally received the promotion she had been hoping for.`, `The sun gently warmed the sandy beach as families laughed and played in the clear blue waves.`, `As the orchestra played the enchanting melody, the audience was captivated by the breathtaking performance of the talented musicians.`."
    # world_prompt = "The sentence is about world news, but not bussiness, science and sports, such as `Amidst the ongoing global challenges, countries around the world are coming together to support refugees and provide aid to those affected by natural disasters and conflicts.`, `In response to the rising humanitarian crisis, international organizations are mobilizing resources to address food insecurity and provide essential medical assistance to vulnerable communities in conflict-affected regions.`."
    # bussiness_prompt = "The sentence is about business news, but not world, science and sports, such as `The tech giant's stock soared to an all-time high after the successful launch of their groundbreaking product that revolutionizes the way we interact with technology.`, `The retail company announced its plans to expand its presence by opening ten new stores across the country, creating hundreds of job opportunities in the process.`."
    # science_prompt = "The sentence is about science news, but not world, bussiness and sports, such as `Scientists have discovered a promising new method for combating antibiotic-resistant bacteria, offering hope for more effective treatments in the future.`, `Researchers have successfully developed a breakthrough in renewable energy technology, significantly improving the efficiency of solar cells and paving the way for a greener and more sustainable future.`." 
    # sports_prompt = "The sentence is about sports news, but not world, science and bussiness, such as `The underdog team staged an incredible comeback in the final minutes of the game, securing a surprising victory against the reigning champions and leaving fans in awe of their determination and skill.`, `In a thrilling match, the star player scored a hat-trick, leading their team to a decisive win and earning praise from fans and critics alike for their outstanding performance.`."
    # postfix_prompt = "\n\n### Sentence:"
    prefix_prompt = 'Please write a sentence satisfies the following constraints.'
    negative_prompt = "The sentence is of negative sentiment, such as `The movie I watched last night was utterly disappointing and boring.`, `The restaurant I visited for dinner tonight served the most tasteless and overpriced food I've ever had.`, `The constant rain and gloomy weather ruined my weekend plans.`"
    positive_prompt = "The sentence is of positive sentiment, such as `After months of hard work and dedication, Sarah finally received the promotion she had been hoping for.`, `The sun gently warmed the sandy beach as families laughed and played in the clear blue waves.`, `As the orchestra played the enchanting melody, the audience was captivated by the breathtaking performance of the talented musicians.`."
    world_prompt = "The sentence is about world news, but not bussiness, science and sports, such as `Amidst the ongoing global challenges, countries around the world are coming together to support refugees and provide aid to those affected by natural disasters and conflicts.`, `In response to the rising humanitarian crisis, international organizations are mobilizing resources to address food insecurity and provide essential medical assistance to vulnerable communities in conflict-affected regions.`."
    bussiness_prompt = "The sentence is about business news, but not world, science and sports, such as `The tech giant's stock soared to an all-time high after the successful launch of their groundbreaking product that revolutionizes the way we interact with technology.`, `The retail company announced its plans to expand its presence by opening ten new stores across the country, creating hundreds of job opportunities in the process.`."
    science_prompt = "The sentence is about science news, but not world, bussiness and sports, such as `Scientists have discovered a promising new method for combating antibiotic-resistant bacteria, offering hope for more effective treatments in the future.`, `Researchers have successfully developed a breakthrough in renewable energy technology, significantly improving the efficiency of solar cells and paving the way for a greener and more sustainable future.`." 
    sports_prompt = "The sentence is about sports news, but not world, science and bussiness, such as `The underdog team staged an incredible comeback in the final minutes of the game, securing a surprising victory against the reigning champions and leaving fans in awe of their determination and skill.`, `In a thrilling match, the star player scored a hat-trick, leading their team to a decisive win and earning praise from fans and critics alike for their outstanding performance.`."
    postfix_prompt = "\n\n### Sentence:"

    def data_collator(self,):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer)

    def preprocess_train(self, data_point):
        has_sent = any('negative' in data_point['prompt'] or 'positive' in data_point['prompt'])
        has_topic = any(t in data_point['prompt'] for t in ['world_prompt','bussiness_prompt','science_prompt', 'sports_prompt'])
        if has_sent and has_topic:
            sent_prompt = data_point['prompt'][0]
            topic_prompt = data_point['prompt'][1]
        elif has_sent:
            sent_prompt = data_point['prompt']
            topic_prompt = random.choice(['world_prompt','bussiness_prompt','science_prompt', 'sports_prompt'])
        elif has_topic:
            sent_prompt = random.choice(['negative_prompt','positive_prompt'])
            topic_prompt = data_point['prompt']
        else:
            raise Exception('')
        user_prompt = self.prefix_prompt + getattr(self,sent_prompt) + getattr(self,topic_prompt) + self.postfix_prompt
        
        # attr_prompt = getattr(self,data_point['prompt'])
        # user_prompt  = self.prefix_prompt + attr_prompt + self.postfix_prompt
        
        ignore_len = (len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len + 1,
        )["input_ids"])- 1) # no eos token
        user_prompt = user_prompt + data_point['output']
        full_tokens = self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # add eos
        ignore_len += max(1, (len(full_tokens) - ignore_len)//2 )
        assert ignore_len < len(full_tokens)
        return {
            "input_ids": full_tokens,
            "labels": [-100] * ignore_len + full_tokens[ignore_len:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def preprocess_gen(self, data_point):
        user_prompt = self.prefix_prompt
        for p in data_point['prompt']:
            attr_prompt = getattr(self,p)
            user_prompt  += attr_prompt 
        user_prompt += self.postfix_prompt 
        if 'input' in data_point:
            user_prompt += data_point['input'] 
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids

    def postprocess(self, text, render=False, split=True):
        output = text.split("\n\n### Sentence:")[-1]
        output = output.replace('�','').replace('</s>','').replace('<s>','')
        return output

class attribute_dialogue():
    prompt_pre = ""
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def __init__(self, tokenizer, decoder_tokenizer, max_len, decoder_add_eos=False, decoder_add_bos=False):
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len
        self.decoder_add_eos=decoder_add_eos
        self.decoder_add_bos=decoder_add_bos

    def process_text(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})
        return {
            "input": user_prompt,
            "output": data_point['output'][-1].strip(),
        }

    def preprocess_train(self, data_point):
        # NOTE is encoder-decoder 
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})

        prior_input_ids = self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # prior not need eos?
        posterior_input_ids = self.tokenizer(
            user_prompt + data_point["output"][-1].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # posterior need eos
        decoder_input_ids = self.decoder_tokenizer(
            data_point["output"][-1].strip(), 
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # TODO: .19 will add bos!! why?
        if self.decoder_add_eos:
            decoder_input_ids += [self.decoder_tokenizer.eos_token_id]
        if self.decoder_add_bos:
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        return {
            "prior_input_ids": prior_input_ids,
            "posterior_input_ids": posterior_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": decoder_input_ids,
            "prior_attention_mask": [1] * (len(prior_input_ids)),
            "posterior_attention_mask": [1] * (len(posterior_input_ids)),
            'tag': data_point.get('tag',None),
        }
    
    def preprocess_train_context(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})

        prior_input_ids = self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # prior not need eos?
        posterior_input_ids = self.tokenizer(
            user_prompt + data_point["output"][-1].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # posterior need eos
        len_input = len(self.decoder_tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) # gpt2 will not add eos token!!
        
        decoder_input_ids = self.decoder_tokenizer(
            user_prompt + data_point["output"][-1].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        if self.decoder_add_eos:
            decoder_input_ids += [self.decoder_tokenizer.eos_token_id]
        if self.decoder_add_bos:
            len_input += 1
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        return {
            "prior_input_ids": prior_input_ids,
            "posterior_input_ids": posterior_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": [-100] * len_input + decoder_input_ids[len_input:],
            "prior_attention_mask": [1] * (len(prior_input_ids)),
            "posterior_attention_mask": [1] * (len(posterior_input_ids)),
            'tag': data_point.get('tag',None),
        }

    def preprocess_gen(self, data_point):
        # TODO ？
        user_prompt = self.prompt_pre
        len_avail = self.max_len - len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        input_prompt = self.prompt_post.format_map({'input':data_point['input']})
        len_avail -= len(self.tokenizer(input_prompt, add_special_tokens=False)['input_ids'])
        lens = len(data_point['history'])
        tokenized_lens = []
        # TODO  | 
        for i in range(lens):
            tmp_prompt = self.prompt_history.format_map(data_point['history'][i])
            tokenized_lens.append(len(self.tokenizer(tmp_prompt,add_special_tokens=False)["input_ids"]))
        
        # ：/2 
        i = 0
        while sum(tokenized_lens) > len_avail and i < lens:
            history = data_point['history'][i]
            tmp_len1 = len(history['input'])
            tmp_len2 = len(history['output'])
            if tmp_len2 > tmp_len1:
                history['output'] = history['output'][:tmp_len2//2]
            else:
                history['input'] = history['input'][:tmp_len1//2]
            prompt = self.prompt_history.format_map(history)
            single_len =(len(self.tokenizer(prompt,add_special_tokens=False)["input_ids"]))
            tokenized_lens[i] = single_len
            i += 1
        total_len = sum(tokenized_lens)
        #  
        while total_len > len_avail and i < lens - 1 :
            total_len -= tokenized_lens[i]
            data_point['history'] = data_point['history'][1:]
            i += 1
        # 
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point['history'][i])
        user_prompt += input_prompt
        inputs = self.tokenizer(user_prompt)["input_ids"]
        decoder_input_ids = self.decoder_tokenizer(
            data_point["output"].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        # NOTE must no eos for generation !!!
        # if self.decoder_add_eos:
        #     decoder_input_ids += [self.decoder_tokenizer.eos_token_id]
        if self.decoder_add_bos:
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        return {
            'input_ids': inputs,
            # 'decoder_input_ids': None,
            'label_ids': decoder_input_ids,
            'attention_mask': [1]*len(inputs),
            'history': user_prompt,
            'label_texts': data_point['output'],
            'tag': data_point.get('tag',None),
        }

    def preprocess_gen_context(self, data_point):
        # TODO ？
        user_prompt = self.prompt_pre
        len_avail = self.max_len - len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        input_prompt = self.prompt_post.format_map({'input':data_point['input']})
        len_avail -= len(self.tokenizer(input_prompt, add_special_tokens=False)['input_ids'])
        lens = len(data_point['history'])
        tokenized_lens = []
        # TODO  | 
        for i in range(lens):
            tmp_prompt = self.prompt_history.format_map(data_point['history'][i])
            tokenized_lens.append(len(self.tokenizer(tmp_prompt,add_special_tokens=False)["input_ids"]))
        
        # ：/2 
        i = 0
        while sum(tokenized_lens) > len_avail and i < lens:
            history = data_point['history'][i]
            tmp_len1 = len(history['input'])
            tmp_len2 = len(history['output'])
            if tmp_len2 > tmp_len1:
                history['output'] = history['output'][:tmp_len2//2]
            else:
                history['input'] = history['input'][:tmp_len1//2]
            prompt = self.prompt_history.format_map(history)
            single_len =(len(self.tokenizer(prompt,add_special_tokens=False)["input_ids"]))
            tokenized_lens[i] = single_len
            i += 1
        total_len = sum(tokenized_lens)
        #  
        while total_len > len_avail and i < lens - 1 :
            total_len -= tokenized_lens[i]
            data_point['history'] = data_point['history'][1:]
            i += 1
        # 
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point['history'][i])
        user_prompt += input_prompt
        inputs = self.tokenizer(user_prompt)["input_ids"]

        decoder_input_ids = self.decoder_tokenizer(
            user_prompt,
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        if self.decoder_add_bos:
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        len_context = len(decoder_input_ids)
        # NOTE must no eos !!!
        # if self.decoder_add_eos:
        #     decoder_input_ids += [self.decoder_tokenizer.eos_token_id]

        full_ids = self.decoder_tokenizer(
            user_prompt + data_point["output"].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        if self.decoder_add_bos:
            full_ids = [self.decoder_tokenizer.bos_token_id] + full_ids
        if self.decoder_add_eos:
            full_ids += [self.decoder_tokenizer.eos_token_id]
        return {
            'input_ids': inputs, # encoder input
            'decoder_input_ids': decoder_input_ids, # decoder context (prefix)
            # ----------------- for forward; can has eos ---------------------
            'full_ids': full_ids,
            'label_ids': [-100]*len_context + full_ids[len_context:], 
            'attention_mask': [1]*len(inputs),
            'history': user_prompt,
            'label_texts': data_point['output'],
            'tag': data_point.get('tag',None),
        }

    def postprocess(self, text, render=False, split=True):
        output = text.split("Assistant:")[-1]
        if split and 'User:' in output:
            output = output.split("User:")[0]
        output = output.replace('�','') 
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def data_collator(self,):
        def collator(features, return_tensors=None):
            # `tokenizer.pad` won't pad labels and must have input_ids and attention_mask
            labels = pad_sequence(
                [ torch.tensor(feature.pop("labels")) for feature in features],
                batch_first=True, 
                padding_value= self.decoder_tokenizer.pad_token_id
            )
            decoder_input_ids = pad_sequence(
                [ torch.tensor(feature.pop("decoder_input_ids")) for feature in features],
                batch_first=True, 
                padding_value= self.decoder_tokenizer.pad_token_id
            )
            tags = None
            if 'tag' in features[0] and features[0]['tag'] is not None:
                tags = torch.tensor([ feature.pop("tag") for feature in features])
            keys = features[0].keys()
            new_features = {}
            for n in keys:
                new_features[n] = pad_sequence(
                    [torch.tensor(feature[n]) for feature in features],
                    batch_first=True, 
                    padding_value= self.tokenizer.pad_token_id
                )
            features = new_features
            features["labels"] = labels
            features["decoder_input_ids"] = decoder_input_ids
            features['tags'] = tags
            return features
        return collator

    def preprocess_split(self, data_point, drop_single=False):

        user_prompt = self.prompt_pre
        len_pre = len(self.tokenizer(
            user_prompt,
            add_special_tokens=False,
        ))
        assert len_pre < self.max_len

        tokenized_lens = []
        for i in range(len(data_point['input'])):
            tmp_prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            single_len =(len(self.tokenizer(
                tmp_prompt,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]))
            # ：/2 inputoutput
            while single_len > self.max_len:
                tmp_len1 = len(data_point['input'][i])
                tmp_len2 = len(data_point['output'][i])
                if tmp_len2 > tmp_len1:
                    data_point['output'][i] = data_point['output'][i][:tmp_len2//2]
                else:
                    data_point['input'][i] = data_point['input'][i][:tmp_len1//2]
                prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
                single_len =(len(self.tokenizer(
                    prompt,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]))
            tokenized_lens.append(single_len)

        num_tokens = len_pre
        left, right = 0,0
        new_turns = []
        while right < len(tokenized_lens):
            
            l = tokenized_lens[right]
            num_tokens += l

            if num_tokens > self.max_len:
                if left == right:
                    right += 1
                new_turns.append({
                    'input': data_point['input'][left:right],
                    'output': data_point['output'][left:right],
                })
                left = right
                num_tokens = len_pre
            else:
                right +=1
        if right > left:
            new_turns.append({
                'input': data_point['input'][left:right],
                'output': data_point['output'][left:right],
            })
        if drop_single:
            new_turns = [d for d in new_turns if len(d['input'])>1]
        if len(new_turns) > 1:
            print(sum(tokenized_lens)+len_pre,[len(new_turns[i]['input']) for i in range(len(new_turns))])
        return new_turns