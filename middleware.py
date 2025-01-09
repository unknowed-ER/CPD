import copy
import copy
import torch
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification, AutoModelForCausalLM
import math
import transformers
import utils
from datasets import Dataset
import tqdm
import sys
from utils import printf
import prompt
import gradio as gr
from transformers.trainer_callback import PrinterCallback
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from typing import Callable, List, Optional, Union
from collections import namedtuple
import numpy as np
import logging
import random
import spacy
import jieba

class InferMiddleWare:
    
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        # , configself.generation_config
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 60,
            "num_beams": 4,
            # NOTE !
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": 1024,
            "min_new_tokens": 25,
            "do_sample": False,
            "repetition_penalty": 1,
            "max_memory": 1024,
            # "bad_words_ids": tokenizer(['<unk>','<s>'], add_special_tokens=False).input_ids,
            # "force_words_ids": tokenizer(['</s>'], add_special_tokens=False).input_ids, is_constraint_gen_mode can only use `is_contrastive_search_gen_mode`
        } 
        if kwargs:
            for n, v in kwargs.items():
                self.generation_config[n] = v
        # TODO:
        self.device = torch.device("cuda:0")
        self.is_encoder_decoder=False

    def infer_plain(
        self,
        text,
    ):
        # No prompts
        input_ids = self.tokenizer(text, return_tensors="pt")['input_ids'].cuda()
        generation_config = GenerationConfig(**self.generation_config)
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=self.generation_config['repetition_penalty'],
        )
        output = generation_output.sequences[0]
        return self.tokenizer.decode(output)

    def infer_turn(
        self,
        inputs,
        history=None,
        prompt_type='chat',
        use_typewriter=False,
        **kwargs,
    ):
        # TODO:  no need for stream generation wrapper
        history = [] if history is None else history
        tokenizer = self.tokenizer
        model=self.model
        max_memory = self.generation_config['max_memory']
        repetition_penalty=self.generation_config["repetition_penalty"]
        generation_config = GenerationConfig(
            **self.generation_config,
            # bad_words_ids=tokenizer(['\n\nUser:','\n\nAssistant:'], add_special_tokens=False).input_ids,
        )        

        data_point = {}
        if prompt_type == 'instruct':
            PROMPT = prompt.instruct_prompt(tokenizer,max_memory)
        elif prompt_type == 'chat':
            PROMPT = prompt.chat_prompt(tokenizer,max_memory)
        elif prompt_type == 'chat with knowledge':
            PROMPT = prompt.chat_context_prompt(tokenizer,max_memory)
            data_point['context'] = kwargs['extra']
        elif prompt_type == 'chat with persona':
            PROMPT = prompt.chat_persona_prompt(tokenizer,max_memory)
            data_point['persona'] = kwargs['extra']
        elif prompt_type == 'attribute_autoencoder':
            PROMPT = prompt.attribute_prompt(tokenizer, max_memory)
            data_point['attribute'] = kwargs['extra']
        elif prompt_type == 'attribute_context':
            PROMPT = prompt.attribute_context(tokenizer, max_memory)
            data_point['prompt'] = kwargs['extra']
        elif prompt_type == 'attribute_chathf_prompt':
            PROMPT = prompt.attribute_chathf_prompt(tokenizer, max_memory)
            data_point['attribute'] = kwargs['extra']
        elif prompt_type == 'raw':
            PROMPT = prompt.prompt(tokenizer, max_memory)
        else:
            raise Exception('no implementation')
        
        data_point['history'] = copy.deepcopy(history)
        data_point['input'] = inputs

        input_ids = PROMPT.preprocess_gen(data_point)
        
        printf('------------------------------')
        printf(tokenizer.decode(input_ids))
        
        input_ids = torch.tensor([input_ids]).to(model.device) # batch=1
        
        printf('------------------------------')
        printf('shape',input_ids.size())
        printf('------------------------------')

        with torch.no_grad():
            if not use_typewriter:
                try:
                    generation_output = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        repetition_penalty=float(repetition_penalty),
                    )
                    s = generation_output.sequences[0]
                    output = tokenizer.decode(s)
                    printf('------------------------------')
                    printf(output)
                    printf('------------------------------')
                    # ，split
                    output = PROMPT.postprocess(output, split=False)
                    history.append({
                        'input': inputs,
                        'output': output,
                    })
                    return output, history
                except torch.cuda.OutOfMemoryError:
                    # avoid exit the program
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    print('>>> Cuda Out of Memory')
                    self.generation_config['max_new_tokens'] //= 2
                    out, his = self.infer_turn(
                        inputs,
                        history,
                        prompt_type,
                        use_typewriter,
                    )
                    self.generation_config['max_new_tokens'] *= 2
                    return out, his
            else:
                try:
                    for generation_output in model.stream_generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                        repetition_penalty=float(repetition_penalty),
                    ):
                        gen_token = generation_output[0][-1].item()
                        # printf(gen_token, end='(')
                        # printf(tokenizer.decode(gen_token), end=') ')
                        printf(tokenizer.decode(gen_token),end=' ')
                        
                        outputs = tokenizer.batch_decode(generation_output) # incase OOM
                        
                except torch.cuda.OutOfMemoryError:
                    print('CUDA out of memory')
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.generation_config['max_new_tokens'] //= 2
                    return self.infer_turn(
                        inputs,
                        history,
                        prompt_type,
                        use_typewriter,
                    )
                
                # finally only one
                printf('[EOS]', end='\n')
                output = PROMPT.postprocess(outputs[0] if outputs is not None else '### Response:')
                return_len = len(output)
                printf(output)
                if return_len > 0:
                    output = PROMPT.postprocess(outputs[0], render=False)
                    history.append({
                        'input': inputs,
                        'output': output,
                    })
                return output, history

    def generate_batched(
        self,
        input_ids: List[torch.Tensor],
        length_sampler= None,
        batch_size: int = 4,
        pad_to_multiple_of: int = None,
        **generation_kwargs,
    ):
        # input: tensor, output: tensor
        self.generation_config.update(generation_kwargs)
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(input_ids), batch_size)

        for i in tqdm.trange(0, len(input_ids), batch_size, desc='generating '):
            if length_sampler is not None:
                self.generation_config["max_new_tokens"] = length_sampler()
            end_index = min(len(input_ids), i + batch_size)

            batch = input_ids[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}
            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.device)

            if 'max_memory' in self.generation_config:
                # TODO: > /home/lzy/miniconda3/envs/trl/lib/python3.10/site-packages/transformers/generation/utils.py(1231)generate()
                # -> self._validate_model_kwargs(model_kwargs.copy())
                self.generation_config.pop('max_memory')
            generations = self.model.generate(**padded_inputs, **self.generation_config)
            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt
                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def infer_gradio(
        self,
        inputs,
        history=None,
        prompt_type='chat',
        use_typewriter=False,
        show_beam=False,
        **kwargs,
    ): 
        # gradio ，yield
        history = [] if history is None else history
        tokenizer = self.tokenizer
        model=self.model
        max_memory = self.generation_config['max_memory']
        repetition_penalty=self.generation_config["repetition_penalty"]
        generation_config = GenerationConfig(
            **self.generation_config,
            # bad_words_ids=tokenizer(['\n\nUser:','\n\nAssistant:'], add_special_tokens=False).input_ids,
        )        
        data_point = {}
        if prompt_type == 'instruct':
            PROMPT = prompt.instruct_prompt(tokenizer,max_memory)
        elif prompt_type == 'chat':
            PROMPT = prompt.chat_prompt(tokenizer,max_memory)
        elif prompt_type == 'chat with knowledge':
            PROMPT = prompt.chat_context_prompt(tokenizer,max_memory)
            data_point['context'] = extra
        elif prompt_type == 'chat with persona':
            PROMPT = prompt.chat_persona_prompt(tokenizer,max_memory)
            data_point['persona'] = extra
        elif prompt_type == 'attribute_autoencoder':
            PROMPT = prompt.attribute_prompt(tokenizer, max_memory)
            data_point['attribute'] = kwargs['extra']
        elif prompt_type == 'raw':
            PROMPT = prompt.prompt(tokenizer, max_memory)
        else:
            raise Exception('not support')
        
        data_point['history'] = copy.deepcopy(history)
        data_point['input'] = inputs

        input_ids = PROMPT.preprocess_gen(data_point)
        
        printf('------------------------------')
        printf(tokenizer.decode(input_ids))
        
        input_ids = torch.tensor([input_ids]).to(model.device) # batch=1
        
        printf('------------------------------')
        printf('shape',input_ids.size())
        printf('------------------------------')

        return_text = [(item['input'], item['output']) for item in history]
        out_memory =False
        outputs = None
        with torch.no_grad():
            #  / 
            # streamly output / typewriter style
            if use_typewriter:
                try:
                    for generation_output in model.stream_generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                        repetition_penalty=float(repetition_penalty),
                    ):
                        gen_token = generation_output[0][-1].item()
                        printf(gen_token, end='(')
                        printf(tokenizer.decode(gen_token), end=') ')
                        
                        outputs = tokenizer.batch_decode(generation_output)
                        if show_beam:
                            show_text = "\n--------------------------------------------\n".join(
                                [ PROMPT.postprocess(output)+" ▌" for output in outputs]
                            )
                        else:
                            show_text = PROMPT.postprocess(outputs[0])+" ▌"
                        if '$' in show_text and gr.__version__ < '3.28':
                            print(f'\x1b[31;20m gradio with version {gr.__version__} cannot show $ correctly!\x1b[0m')
                            show_text = show_text.replace('$', '`')
                        yield return_text +[(inputs, show_text)], history
                except torch.cuda.OutOfMemoryError:
                    print('CUDA out of memory')
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    out_memory=True
                
                # finally only one
                printf('[EOS]', end='\n')
                show_text = PROMPT.postprocess(outputs[0] if outputs is not None else '### Response:')
                return_len = len(show_text)
                if out_memory==True:
                    out_memory=False
                    show_text+= '<p style="color:#FF0000"> [GPU Out Of Memory] </p> '
                if return_len > 0:
                    output = PROMPT.postprocess(outputs[0], render=False)
                    history.append({
                        'input': inputs,
                        'output': output,
                    })
                printf(show_text)
                return_text += [(inputs, show_text)]
                yield return_text, history
            # common 
            else:
                try:
                    generation_output = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=float(repetition_penalty),
                    )
                    s = generation_output.sequences[0]
                    output = tokenizer.decode(s)
                    output = PROMPT.postprocess(output)
                    history.append({
                        'input': inputs,
                        'output': output,
                    })
                    return_text += [(inputs, output)]
                    yield return_text, history
                except torch.cuda.OutOfMemoryError:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    show_text = '<p style="color:#FF0000"> [GPU Out Of Memory] </p> '
                    printf(show_text)
                    return_text += [(inputs, show_text)]
                    yield return_text, history

    def infer_chat(
        self,
        data,
        prompt_type='chat',
        use_typewriter=False,
        **kargs
    ):
        ans = []
        for line in tqdm.tqdm(data):
            history = []
            for turn in line['history']:
                outs, history = self.infer_turn(
                    turn['input'],
                    history,
                    prompt_type,
                    use_typewriter,
                )
            ans.append(history)
        return ans

class TextClassifierMiddleWare:

    def __init__(self, classifier_dict, device='cuda:0', **kwargs):
        # pre_load classifer
        self.classifier_dict= classifier_dict
        self.device= device

    def compute_metrics(self, pred):
        labels = torch.tensor(pred.label_ids).long()
        preds = torch.softmax(torch.tensor(pred.predictions,dtype=float),dim=-1)
        # out[i][j] = preds[i][labels[i][j]]
        probs = torch.gather(preds, 1, labels.view(-1, 1))
        acc = torch.mean(probs).item()
        return {
            # NOTE ，bad case
            'scores': [prob.item() for prob in probs],
            'accuracy': round(acc,6)
        }
    
    def make_batch(self, sents, labels, name, sents2=None):
        # label {'labels': , 'sent': }
        
        tokenizer=self.classifier_dict[name]['tokenizer'] 
        dataset = Dataset.from_dict({
            'labels': labels,
            'text': sents,
        })
        if isinstance(name, str) and 'agnews-topic' in name:
            # TODO: change to sents2
            # 
            topics = ["world","sports","business","science"]
            eval_dataset = dataset.map(lambda e: tokenizer(topics[e['labels']]+'[SEP]'+e['text'], truncation=True, padding='max_length', max_length=100))
            eval_dataset = eval_dataset.map(lambda e: {'labels': 1})
        elif name == 'nli':
            dataset = Dataset.from_dict({
                'input': sents,
                'output': sents2,
                'labels': labels,
            })
            eval_dataset = dataset.map(lambda example: tokenizer(
                example['input'] + tokenizer.sep_token + example['output'], 
                truncation=True, 
                padding=False, 
                max_length=512,
            ))
        else:
            eval_dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return eval_dataset

    def test_single(self, texts, labels, name, reduce_sum=True, texts2=None):
        eval_dataset = self.make_batch(texts, labels, name, texts2)

        # research
        if 'model' not in self.classifier_dict[name]:
            model = self.classifier_dict[name]['text']
        else:
            model = self.classifier_dict[name]['model']
        # avoid evaluate log
        trainer = transformers.Trainer(
            model=model,
            args = transformers.TrainingArguments(
                output_dir='outs',
                do_train = False,
                disable_tqdm=True,
                do_predict = True,
                per_device_eval_batch_size=8 if 'name' in ['nli'] else 128,
                dataloader_drop_last = False,
                report_to=[]
            ),
            compute_metrics=self.compute_metrics,
            data_collator=transformers.DataCollatorWithPadding(self.classifier_dict[name]['tokenizer'] ),
        )
        trainer.remove_callback(PrinterCallback)
        if reduce_sum:
            return trainer.evaluate(eval_dataset)['eval_accuracy']
        return {
            'scores': trainer.evaluate(eval_dataset)['eval_scores'],
            'acc': trainer.evaluate(eval_dataset)['eval_accuracy'],
        }

    def compute_acc(self, texts, labels, names, reduce_sum=True):
        # ：label
        result = []
        assert len(labels) == len(names)
        for label, name in zip(labels, names):
            result.append( self.test_single(
                texts,
                [label]*len(texts),
                name,
                reduce_sum,
            ))
        if reduce_sum:
            return {n:v for n,v in zip(names, result)}, None
        acc = {n:v['acc'] for n,v in zip(names, result)}
        acc_details = [ r['scores'] for r in result ]
        acc_details = [ list(d) for d in zip(*acc_details)]
        acc_details = [ {n:dd for dd, n in zip(d,names)} for d in acc_details]
        return acc , acc_details
            
    def compute_acc2(self, texts, labels, name, reduce_sum=True):
        # label
        assert len(texts) == len(labels)
        if isinstance(name, list):
            result = []
            for i, n in enumerate(name):
                result.append( self.test_single(
                    texts,
                    [label[i] for label in labels],
                    n,
                    reduce_sum,
                ))
            if reduce_sum:
                return {n:v for n,v in zip(name, result)}, None
            acc = {n:v['acc'] for n,v in zip(name, result)}
            acc_details = [ r['scores'] for r in result ]
            acc_details = [ list(d) for d in zip(*acc_details)]
            acc_details = [ {n:dd for dd, n in zip(d,name)} for d in acc_details]
            return acc , acc_details
        else:
            result =  self.test_single(
                texts,
                labels,
                name,
                reduce_sum,
            )
            if reduce_sum:
                return result, None
            return result['acc'], result['scores']

    def compute_NLI(self, history, response, reduce_sum=True):
        assert len(history) == len(response)
        result =  self.test_single(
            history,
            [1]*len(history),
            'nli', 
            reduce_sum,
            texts2=response,
        )
        if reduce_sum:
            return result, None
        return result['acc'], result['scores']

class GenerationMetricMiddleWare:

    def compute_bleu(self, label, pred, language='zh'):
        # sentence bleu  corpus bleu ： https://zhuanlan.zhihu.com/p/404381278
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import numpy as np
        score = 0
        result = {
            'sentence_bleu': 0,
            'corpus_bleu': 0,
            'bleu4': 0,
            'bleu3': 0,
            'bleu2': 0,
            'bleu1': 0
        }
        if language == 'zh':
            for key, weights in zip(['bleu4', 'bleu3', 'bleu2', 'bleu1'],
                                    [
                                        (0.25, 0.25, 0.25, 0.25),  #  bleu4
                                        (0.33, 0.33, 0.33, 0),  # 3
                                        (0.5, 0.5, 0, 0),  # 2
                                        (1, 0, 0, 0),  # 1
                                    ]):
                blue_score = np.mean(
                    [sentence_bleu(
                        references=[list(a)],
                        hypothesis=list(b),
                        smoothing_function=SmoothingFunction().method1,
                        weights=weights
                    ) for a, b in zip(label, pred)]
                )
                score += blue_score
                result[key] = blue_score
        else:
            for key, weights in zip(['bleu4', 'bleu3', 'bleu2', 'bleu1'],
                                    [
                                        (0.25, 0.25, 0.25, 0.25),  #  bleu4
                                        (0.33, 0.33, 0.33, 0),  # 3
                                        (0.5, 0.5, 0, 0),  # 2
                                        (1, 0, 0, 0),  # 1
                                    ]):
                blue_score = np.mean(
                    [sentence_bleu(
                        references=[a.split()],
                        hypothesis=list(b.split()),
                        smoothing_function=SmoothingFunction().method1,
                        weights=weights
                    ) for a, b in zip(label, pred)]
                )
                score += blue_score
                result[key] = blue_score

        sentence_score = score/4
        corpus_score = corpus_bleu([[t] for t in pred], label)
        result['sentence_bleu'] = float(sentence_score)
        result['corpus_bleu'] = float(corpus_score)
        return result

    def compute_lawrouge(self,label,pred):
        import lawrouge
        rouge = lawrouge.Rouge()
        # [。。]
        rouge.sentence_split = None
        # ['']
        for i,p in enumerate(pred):
            if p == '':
                pred[i] = '。'
        scores = rouge.get_scores(self, pred,label, avg=1)
        return scores

    def compute_rouge2(self,label, pred, weights=None, mode='weighted'):
        # Problem: cannot handle '.....' and ''
        import rouge
        rouge = rouge.Rouge()
        weights = weights or (0.2, 0.4, 0.4)
        if isinstance(label, str):
            label = [label]
        if isinstance(pred, str):
            pred = [pred]  
        # ，0;
        # label = [' '.join(x) for x in label]
        # pred = [' '.join(x) for x in pred]
        pred = [ 'a' if l=='' or l=='.' else l for l in pred]
        rouge_dict = rouge.get_scores(hyps=pred, refs=label, avg=1)
        rouge_dict = {k: 100* v['f'] for k,v in rouge_dict.items()}
        return rouge_dict

    def compute_rouge(self, label, pred):
        from rouge_score import rouge_scorer
        rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
        rouge_score = {k:0 for k in rouge_metrics}
        scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)
        for l ,p in zip(label, pred):
            rouge_score = { k: rouge_score[k]+ v.fmeasure for k,v in scorer.score(target=l, prediction=p).items() }
        rouge_score = {k: v/len(pred) for k,v in rouge_score.items()}
        return rouge_score

    def compute_bertscore(self,label,pred):
        import bert_score
        P, R, F1 = bert_score.score(pred, label, lang="zh", verbose=True)
        return F1.mean()

    def compute_distinct4(args, tokenized_gen, language='zh'):
        import re
        dis1 = 0
        dis_set1 = set()
        dis2 = 0
        dis_set2 = set()
        for pred in tokenized_gen:

            if language == 'zh':
                pred = list(jieba.cut(pred))
            else:
                pred = pred.split(' ')
            for word in pred:
                dis_set1.add(word)
                dis1 += 1
            for i in range(1, len(pred)):
                dis_set2.add(pred[i - 1] + ' ' + pred[i])
                dis2 += 1
        return {'distinct1':len(dis_set1) / dis1, 'distinct2': len(dis_set2) / dis2}

    def compute_distinct(self, responses):
        # len unique ngram / len words 
        # compute in corpus level
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in responses:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1=len(unigrams) / total_words
        dist2=len(bigrams) / total_words
        dist3=len(trigrams) / total_words
        # take the mean across prompts
        metrics = {"dist1":float(np.nanmean(dist1)),"dist2":float(np.nanmean(dist2)),"dist3":float(np.nanmean(dist3))}
        return metrics

    def compute_distinct2(self, responses):
        # len unique ngram / len words 
        # compute in sentence level 
        dist1, dist2, dist3 = [],[],[] 
        total_words = 0
        for gen in responses:
            o = gen.split(' ')
            total_words = len(o)
            unigrams, bigrams, trigrams = set(), set(), set()
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
            dist1.append(len(unigrams) / total_words)
            dist2.append(len(bigrams) / total_words)
            dist3.append(len(trigrams) / total_words)
        # take the mean across prompts
        metrics = {"dist1":float(np.nanmean(dist1)),"dist2":float(np.nanmean(dist2)),"dist3":float(np.nanmean(dist3))}
        return metrics

    @torch.inference_mode()
    def compute_ppl2(self, preds, prompts=None, model_name=f'{utils.MODEL_DIR}/gpt2-xl', reduce_sum=True, gpu=0):
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = "False"
        preds = [p for p in preds if p != '']
        device = "cuda:" + str(gpu) if torch.cuda.is_available() else "cpu"
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": gpu},
        )
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        perplexities = []
        if prompts is None:
            prompts = [None]*len(preds)
        for pred, prompt in tqdm.tqdm(zip(preds, prompts),total=len(preds), desc='PPL ', leave=False):
            if prompt is not None:
                len_user_pred_tokens = len(tokenizer(pred)["input_ids"][:-1])
                pred_label = tokenizer(pred, truncation=True, padding=False, return_tensors='pt', max_length=4096)['input_ids']
                full_input = prompt + pred
                input_ids = tokenizer(full_input, truncation=True, padding=False, return_tensors='pt', max_length=4096).to(device=device)
                result = model(**input_ids).logits[:, -len_user_pred_tokens:, :]
                loss = torch.nn.functional.nll_loss(result.squeeze(0).cpu(), pred_label[:, :-1].squeeze(0).cpu())
                ppl = math.exp(loss.item())
            else:
                full_input_ids = tokenizer.encode(pred, return_tensors='pt').to(device)
                full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1] - 1)
                loss = full_loss / full_input_ids.shape[1]
                ppl = math.exp(loss.item())

            if ppl < 1e4:   # for sanity
                perplexities.append(ppl)
        del model
        del tokenizer
        if reduce_sum:
            return np.nanmean(perplexities)
        else:
            return perplexities

    def compute_ppl(self, pred):
        # ~/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--perplexity/8ab643ad86f568b7d1d5f7822373fa7401ff5ff0297ccf114b0ca6a33be96bc0/perplexity.py"
        if '' in pred: # len must >= 1
            return 0
        perplexity = evaluate.load("perplexity", module_type="metric")
        results = perplexity.compute(predictions=pred, model_id=f'{utils.MODEL_DIR}/gpt2-xl')
        return results['mean_perplexity']

    def compute_sbleu(self,texts):
        senbleu = 0 
        texts = [text.split(' ') for text in texts]
        for i in tqdm.trange(len(texts), desc='calc senblue',leave=False):
            # ref, hyp
            # texts.pop(i) texts
            senbleu += sentence_bleu(
                references=random.sample(texts[:i]+texts[i+1:],150), 
                hypothesis=texts[i], 
                smoothing_function=SmoothingFunction().method1
            )
        senbleu = senbleu/len(texts)
        return senbleu

    def compute_sbleu_fast(self,texts):
        import multiprocess
        from multiprocess import Pool
        from queue import Empty
        import random
        import os
        # important , or else get 1. but split list will cause the process extremely slow
        texts = [text.split(' ') for text in texts]
        def _sbleu(i):
            return sentence_bleu(
                references=random.sample(texts[:i]+texts[i+1:],150), 
                hypothesis=texts[i], 
                smoothing_function=SmoothingFunction().method1
            )

        sbleus = []
        # with Pool(os.cpu_count()) as pool:
        #     sbleus = list((pool.map(_sbleu, range(len(texts))), total=len(texts), desc='sbleu'))
        # sbleu = utils.mean(sbleus)

PERSPECTIVE_API_KEY = 'AIzaSyAJRwCXotWADJqqjUbEs_MKCTZG0e7-2fw'
PERSPECTIVE_API_ATTRIBUTES = {'TOXICITY'}
class PerspectiveAPI:
    import time
    from typing import List, Union, Optional, Tuple, Dict, Any, Iterable
    
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY, rate_limit: int = 256):
        self.service = self._make_service(api_key)
        self.last_request_time = -1  # satisfies initial condition
        self.rate_limit = rate_limit
        self.next_uid = 0

    def request(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]

        # Rate limit to 1 batch request per second
        assert len(texts) <= self.rate_limit
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        self.last_request_time = time.time()

        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {str(uid): None for uid in range(self.next_uid, self.next_uid + len(texts))}
        self.next_uid += len(texts)

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Make API request
        batch_request = self.service.new_batch_http_request()
        for uid, text in zip(responses.keys(), texts):
            batch_request.add(self._make_request(text, self.service), callback=response_callback, request_id=uid)
        batch_request.execute()
        responses = list(responses.values())
        output = [{'text': i,'toxicity': j[0]['attributeScores']['TOXICITY']['summaryScore']['value']}for i,j in zip(texts,responses) if j[0] is not None ]
        return output
    
    @staticmethod
    def _make_service(api_key: str):
        from googleapiclient import discovery # google-api-python-client==2.36.0
        # Generate API client object dynamically based on service name and version
        return discovery.build('comments:analyze', 'v1alpha1',
                               discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                               developerKey=api_key,
                               static_discovery=False)

    @staticmethod
    def _make_request(text: str, service):
        analyze_request = {
            'comment': {'text': text},
            'languages': ['en'],
            'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
            'spanAnnotations': True,
        }
        return service.comments().analyze(body=analyze_request)

class TenseModel:
    def __init__(self, tense_segments: list):
        self.segments = tense_segments
        self.tense = None
        self.find_tense(self.get_passive_pattern())
        if self.tense is None:
            self.find_tense(self.get_normal_pattern())

    @staticmethod
    def get_passive_pattern():
        return {
            'PassiveSimplePresent': [{'be': ['VBZ', 'VBP']}, {'_': ['VBN']}],
            'PassiveSimplePresent_One': [{'_all_modal_': ['MD']}, {'be': ['VB']}, {'_': ['VBN']}],
            'PassivePresentContinuous': [{'be': ['VBZ', 'VBP']}, {'be': ['VBG']}, {'_': ['VBN']}],
            'PassiveSimplePast': [{'be': ['VBD']}, {'_': ['VBN']}],
            'PassivePastContinuous': [{'be': ['VBD']}, {'be': ['VBG']}, {'_': ['VBN']}],
            'PassivePresentPrefect': [{'have': ['VBZ', 'VBP']}, {'be': ['VBN']}, {'_': ['VBN']}],
            'PassivePastPrefect': [{'have': ['VBD']}, {'be': ['VBN']}, {'_': ['VBN']}],
            'PassiveFuture': [{'will': ['MD']}, {'be': ['VB']}, {'_': ['VBN']}],
            'PassiveFutureContinuous': [{'will': ['MD']}, {'be': ['VB']}, {'be': ['VBG']}, {'_': ['VBN']}],
            'PassivePresentConditional': [{'would': ['MD']}, {'be': ['VB']}, {'_': ['VBN']}],
            'PassivePastConditional': [{'would': ['MD']}, {'have': ['VB']}, {'be': ['VBN']}, {'_': ['VBN']}],
            'PassiveInfinitive': [{'must': ['MD']}, {'be': ['VB']}, {'_': ['VBN']}],
        }

    @staticmethod
    def get_normal_pattern():
        return {
            'SimplePresent': [{'_': ['VBZ', 'VBP', 'VB']}],
            'SimplePresentModal_One': [{'have': ['VBZ', 'VBP']}],
            'SimplePresentModal_Two': [{'_all_modal_': ['MD']}, {'_': ['VB']}],
            'PresentContinuous': [{'be': ['VBZ', 'VBP']}, {'_': ['VBG']}],
            'SimplePast': [{'_': ['VBD']}],
            'PastContinuous': [{'be': ['VBD']}, {'_': ['VBG']}],
            'PresentPerfect': [{'have': ['VBZ', 'VBP']}, {'_': ['VBN']}],
            'PresentPerfectContinuous': [{'have': ['VBZ', 'VBP']}, {'be': ['VBN']}, {'_': ['VBG']}],
            'PastPerfect': [{'have': ['VBD']}, {'_': ['VBN']}],
            'PastPerfectContinuous': [{'have': ['VBD']}, {'be': ['VBN']}, {'_': ['VBG']}],
            'FuturePerfect': [{'will': ['MD']}, {'have': ['VB']}, {'_': ['VBN']}],
            'FuturePerfectContinuous': [{'will': ['MD']}, {'have': ['VB']}, {'be': ['VBN']}, {'_': ['VBG']}],
            'SimpleFuture': [{'will': ['MD']}, {'_': ['VB']}],
            'FutureContinuous': [{'will': ['MD']}, {'be': ['VB']}, {'_': ['VBG']}],
        }

    @staticmethod
    def get_all_modal():
        return ['can', 'could', 'may', 'might', 'shall', 'should']

    def find_tense(self, patterns):
        for tense_name, tense_pattern in patterns.items():
            if len(tense_pattern) != len(self.segments):
                continue

            for index, segment_pattern in enumerate(tense_pattern):
                if len(self.segments) > index and self.segments[index] is not None:
                    tense_segment = self.segments[index]
                    if '_' in segment_pattern and len(self.segments) == index + 1:
                        if tense_segment.tag in segment_pattern['_']:
                            self.tense = tense_name
                            return True
                    if tense_segment.lemma in segment_pattern:
                        if tense_segment.tag in segment_pattern[tense_segment.lemma]:
                            if tense_segment.lemma in segment_pattern:
                                continue

                    elif tense_segment.lemma in self.get_all_modal():
                        if '_all_modal_' in segment_pattern:
                            continue
                    elif tense_segment.lemma == 'have' and len(self.segments) == 1:
                        self.tense = tense_name
                        return True
                    break
        return False

class TenseScorer:

    MAP_TABLE = {
        'PassiveSimplePresent': 'present',
        'PassiveSimplePresent_One': 'present',
        'PassivePresentContinuous': 'present',
        'PassiveSimplePast': 'past',
        'PassivePastContinuous': 'past',
        'PassivePresentPrefect': 'present',
        'PassivePastPrefect': 'past',
        'PassiveFuture': 'future',
        'PassiveFutureContinuous': 'future',
        'PassivePresentConditional': 'present',
        'PassivePastConditional': 'past',
        'PassiveInfinitive': 'past',
        'SimplePresent': 'present',
        'SimplePresentModal_One': 'present',
        'SimplePresentModal_Two': 'present',
        'PresentContinuous': 'present',
        'SimplePast': 'past',
        'PastContinuous': 'present',
        'PresentPerfect': 'present',
        'PresentPerfectContinuous': 'present',
        'PastPerfect': 'past',
        'PastPerfectContinuous': 'past',
        'FuturePerfect': 'future',
        'FuturePerfectContinuous': 'future',
        'SimpleFuture': 'future',
        'FutureContinuous': 'future',
    }
    SegmentModel= namedtuple('SegmentModel', 'text lemma tag')

    def __init__(self,  **kwargs):
        self.nlp = spacy.load('en_core_web_sm')

    def findtense(self, text):
        doc = self.nlp(text)
        tense_list = []
        sent_list = []
        for sentence in doc.sents:
            if len(sentence) <= 1:
                continue
            result = []
            current_tense = []
            increment_counter = 0
            for token in sentence:
                if token.pos_ in ('AUX', 'VERB'):
                    increment_counter += 1
                    tense_segment = self.SegmentModel(token.text, token.lemma_, token.tag_)
                    current_tense.append(tense_segment)
                elif increment_counter > 0:
                    increment_counter = 0
                    tense = TenseModel(current_tense)
                    result.append(tense)
                    current_tense = []
            sent_list.append(sentence.text)
            tense_list.append([tense.tense for tense in result])
        if len(sent_list):
            return self.convert_format(sent_list, tense_list)
        else:
            import pdb; pdb.set_trace()
    
    def convert_format(self, sentences, tenses):
        data = []
        tenses = [[ self.MAP_TABLE[tt] if tt is not None else None for tt in t ] if len(t) else [] for t in tenses ]
        tenses = [ ['past', 'present', 'future'][ np.argmax([t.count(s) for s in ['past', 'present', 'future']]) ] if len(t) else [] for t in tenses ]
        final_tense = ['past', 'present', 'future'][ np.argmax([ tenses.count(s) for s in ['past', 'present', 'future'] ]) ]
        flag = False
        for s,t in zip(sentences, tenses):
            if t == [] or t == final_tense:
                if not flag:
                    flag = True
                    data.append([])
                data[-1].append(s)
            elif flag:
                flag = False
        idx = np.argmax([len(d) for d in data])
        return {
            # 'text': ' '.join(data[idx]),
            'tense': final_tense,
        }

    def __call__(self, texts, labels):
        
        acc, res = 0, []
        if isinstance(labels, list): 
            assert len(texts) == len(labels)
            for t, l in zip(texts, labels):
                tense = self.findtense(t)['tense']
                acc += tense == l
                res.append(tense)
            acc /= len(texts)
        else:
            for t in texts:
                tense = self.findtense(t)['tense']
                acc += tense == labels
                res.append(tense)
            acc /= len(texts)
        return acc, res     

class KeywordScorer:
    def __init__(self, **kwargs) -> None:
        pass
    
    #  ' ' 
    # ： text: I am happy.  label: 'I happy'
    def __call__(self, texts, labels):
        acc , res = 0, []
        if isinstance(labels, list): 
            assert len(texts) == len(labels)
            for t, l in zip(texts, labels):
                cnt = 1
                for ll in l.split(' '):
                    if ll.lower() not in t.lower():
                        cnt = 0
                acc += cnt
                res.append(cnt)
        else:
            for t in texts:
                cnt = 1
                for ll in labels.split(' '):
                    if ll.lower() not in t.lower().split(' '):
                        cnt = 0
                acc += cnt
                res.append(cnt)

        acc /= len(texts)
        return acc, res

class ClassifierScorer:

    def __init__(self, classifier, tokenizer):
        self.tokenizer = tokenizer
        self.model = classifier

    def compute_metrics(self, pred):
        labels = torch.tensor(pred.label_ids).long()
        preds = torch.softmax(torch.tensor(pred.predictions,dtype=float),dim=-1)
        # out[i][j] = preds[i][labels[i][j]]
        probs = torch.gather(preds, 1, labels.view(-1, 1))
        acc = torch.mean(probs).item()
        return {
            # NOTE ，bad case
            'details': [prob.item() for prob in probs],
            'accuracy': round(acc,6)
        }
    
    def make_batch(self, sents, labels):
        # label {'labels': , 'sent': }
        tokenizer=self.tokenizer
        dataset = Dataset.from_dict({
            'labels': labels if isinstance(labels,list) else [labels]*len(sents),
            'text': sents,
        })
        eval_dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return eval_dataset

    def __call__(self, texts, labels):
        eval_dataset = self.make_batch(texts, labels)
        # avoid evaluate log
        trainer = transformers.Trainer(
            model=self.model,
            args = transformers.TrainingArguments(
                output_dir='outs',
                do_train = False,
                disable_tqdm=True,
                do_predict = True,
                per_device_eval_batch_size=8,
                dataloader_drop_last = False,
                report_to=[]
            ),
            compute_metrics=self.compute_metrics,
            data_collator=transformers.DataCollatorWithPadding(self.tokenizer),
        )
        trainer.remove_callback(PrinterCallback)
        ans = trainer.evaluate(eval_dataset)
        return ans['eval_accuracy'], ans['eval_details']

