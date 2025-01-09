import os
import json
import argparse
import sys
import torch
import math
import warnings
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, \
    LlamaTokenizer

import utils
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM
import prompt

'''ddp'''
import torch.distributed as dist
import torch.multiprocessing as mp


def collate_fn(batch):
    if "entities" in batch[0].keys():
        if 'pred_entity' in batch[0].keys():
            instruction, input, output, input_ids, attention_mask, entities, labels, select_entity, pred_entity, kind, dialogue_id, ppl = [], [], [], [], [], [], [], [], [], [], [], []
            for data in batch:
                input_ids.append(data['input_ids'])
                attention_mask.append(data['attention_mask'])
                entities.append(data['entities'])
                instruction.append(data['instruction'])
                input.append(data['input'])
                output.append(data['output'])
                labels.append(data['labels'])
                dialogue_id.append(data['dialogue_id'])
                select_entity.append(data['select_entity'])
                pred_entity.append(data["pred_entity"])
                kind.append(data["kind"])
                ppl.append(data["ppl"] if 'ppl' in data.keys() else 0)


            max_input_len = max([len(input_id) for input_id in input_ids])
            for i in range(len(batch)):
                input_ids[i] = input_ids[i] + [0] * (max_input_len - len(input_ids[i]))
                attention_mask[i] = attention_mask[i] + [0] * (max_input_len - len(attention_mask[i]))
                labels[i] = labels[i] + [-100] * (max_input_len - len(labels[i]))
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            return {
                "instruction": instruction,
                "input": input,
                "output": output,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entities': entities,
                'select_entity': select_entity,
                'pred_entity': pred_entity,
                'kind': kind,
                'ppl': ppl,
                "dialogue_id": dialogue_id,
                'labels': labels
            }
        else:
            instruction, input, output, input_ids, attention_mask, entities, labels, select_entity, dialogue_id = [], [], [], [], [], [], [], [], []
            for data in batch:
                input_ids.append(data['input_ids'])
                attention_mask.append(data['attention_mask'])
                entities.append(data['entities'])
                instruction.append(data['instruction'])
                input.append(data['input'])
                output.append(data['output'])
                labels.append(data['labels'])
                dialogue_id.append(data['dialogue_id'])
                select_entity.append(data['select_entity'])

            max_input_len = max([len(input_id) for input_id in input_ids])
            for i in range(len(batch)):
                input_ids[i] = input_ids[i] + [0] * (max_input_len-len(input_ids[i]))
                attention_mask[i] = attention_mask[i] + [0] * (max_input_len-len(attention_mask[i]))
                labels[i] = labels[i] + [-100] * (max_input_len-len(labels[i]))
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            return {
                "instruction": instruction,
                "input": input,
                "output": output,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entities': entities,
                'select_entity': select_entity,
                "dialogue_id": dialogue_id,
                'labels': labels
            }
    else:
        instruction, input, output, input_ids, attention_mask, labels, select_entity, dialogue_id = [], [], [], [], [], [], [], []
        for data in batch:
            input_ids.append(data['input_ids'])
            attention_mask.append(data['attention_mask'])
            instruction.append(data['instruction'])
            input.append(data['input'])
            output.append(data['output'])
            labels.append(data['labels'])
            dialogue_id.append(data['dialogue_id'])
            select_entity.append(data['select_entity'])

        max_input_len = max([len(input_id) for input_id in input_ids])
        for i in range(len(batch)):
            input_ids[i] = input_ids[i] + [0] * (max_input_len-len(input_ids[i]))
            attention_mask[i] = attention_mask[i] + [0] * (max_input_len-len(attention_mask[i]))
            labels[i] = labels[i] + [-100] * (max_input_len-len(labels[i]))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "instruction": instruction,
            "input": input,
            "output": output,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'select_entity': select_entity,
            "dialogue_id": dialogue_id,
            'labels': labels
        }

@torch.inference_mode()
def compute_ppl(input_data, model):
    loss_func = torch.nn.CrossEntropyLoss()
    batch_size, seq_len = input_data["input_ids"].size()
    input_ids = input_data["input_ids"].to(model.device)
    attention_mask = input_data["attention_mask"].to(model.device)
    labels = input_data["labels"]
    # result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    result = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()
    ppls = []
    for i in range(batch_size):
        try:
            padding_tokens_num = attention_mask[i].eq(0).sum().tolist()
            init_generation_num = input_data["labels"][0].eq(-100).sum() - padding_tokens_num
            pred_porbs = result[i, init_generation_num-1:seq_len-padding_tokens_num-1, :]  # the logits need to shift 1 position
            gold_label = labels[i, init_generation_num:seq_len-padding_tokens_num]
            loss = loss_func(pred_porbs, gold_label)
            ppl = math.exp(loss.item())
            ppls.append(ppl)
        except:
            print('Pred Probs Error! Dialog id is :', input_data['dialogue_id'][i])
            print('Pred Probs Error! Select entity is :', input_data['select_entity'][i])
            ppls.append(ppls[-1])
    return ppls

@torch.inference_mode()
def eval_ppl(local_gpu_rank, args):
    world_size = args.gpus_num
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_gpu_rank)

    if 'lama' in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True, trust_remote_code=True)
        # assert train_tokenizer.eos_token_id == 2, "Tokenizer eos is wrong!!!"
        tokenizer.pad_token_id = 0
    elif 'Qwen' in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, pad_token='<|endoftext|>', eos_token='<|im_end|>', padding_side='left',add_eos_token=True, trust_remote_code=True)

    if local_gpu_rank == 0:
        logger = utils.set_file_logger('get_causal_data', os.path.dirname(args.output_path), True)
        logger.info(f'>>>input path: {args.data_path}')
        logger.info(f'>>>output path: {args.output_path}')
        logger.info(f'>>>model path: {args.model_path}')
        logger.info(f'>>>lora path: {args.lora_path}')

    num_proc = (os.cpu_count()) // world_size
    datasets = load_dataset('json', data_files=args.data_path)
    datasets['train'] = datasets['train'].add_column("dialogue_id", list(range(len(datasets['train']))))
    # datasets['train'] = Dataset.from_dict(datasets['train'][:1])
    # datasets = datasets['train'].train_test_split(train_size=0.005, seed=42)
    PROMPT = prompt.persona_prompt(tokenizer, args.cutoff_len, drop=args.drop, use_data_instruction=False)
    if local_gpu_rank == 0:
        import random;
        start = random.randint(0, len(datasets['train'])-1)
        examples = PROMPT.preprocess_drop(datasets['train'][start:start + 1])
        start = random.randint(1, len(examples['input_ids'])-1)
        logger.info(f'>>> prompt example: {tokenizer.decode(examples["input_ids"][start])}')
        logger.info(f'>>> select entity: {examples["select_entity"][start]}')
        logger.info(f'>>> process dataset:')
    num_proc = min(num_proc, len(datasets['train']))
    processed_dataset = datasets['train'].map(PROMPT.preprocess_drop, num_proc=num_proc, batched=True, remove_columns = datasets['train'].column_names, batch_size=1)  # ，map， batched=True, remove_columns = datasets['train'].column_names, batch_size=1
    if world_size > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(processed_dataset, shuffle=False)
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, sampler=test_sampler)
        test_sampler.set_epoch(0)
    else:
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_has_fp16_weight=False,
    )
    if 'no_pos' in args.lora_path:
        if 'lama' in args.model_path:
            from model import MyLlamaForCausalLM
            model = MyLlamaForCausalLM.from_pretrained(
                args.model_path,
                device_map={"": local_gpu_rank},
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_4bit=True,
                use_flash_attention_2=False,
                quantization_config=bnb_config,
            )
        elif 'Qwen' in args.model_path:
            from model import MyQWenLMHeadModel
            model = MyQWenLMHeadModel.from_pretrained(
                args.model_path,
                device_map={"": local_gpu_rank},
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_4bit=True,
                use_flash_attn=False,
                use_flash_attention_2=False,
                quantization_config=bnb_config,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": local_gpu_rank},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=True,
            use_flash_attention_2=False,
            quantization_config=bnb_config,
        )

    if 'Qwen' in args.model_path:
        for module in model.parameters():
            if hasattr(module, "dtype") and module.dtype == torch.float32:
                if not isinstance(module, torch.Tensor):
                    module.to(torch.bfloat16)
                else:
                    module.data = module.data.to(torch.bfloat16)

    if len(args.lora_path) > 0:
        # fix the path for local checkpoint
        lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
        if not os.path.exists(lora_bin_path):
            pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
            if os.path.exists(pytorch_bin_path):
                os.rename(pytorch_bin_path, lora_bin_path)
                warnings.warn(
                    "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
                )
            else:
                assert ('Checkpoint is not Found!')

        model = StreamPeftGenerationMixin.from_pretrained(
            model, args.lora_path, torch_dtype=torch.float16, device_map={"": local_gpu_rank}
        )

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    output_data_list = []
    for input_data in tqdm(test_dataset):
        ppls = compute_ppl(input_data, model)
        for i in range(len(ppls)):
            output_data = {
                'instruction': input_data['instruction'][i],
                'input': input_data['input'][i],
                'output': input_data['output'][i],
                'entities': input_data['entities'][i],
                'select_entity': input_data['select_entity'][i],
                "dialogue_id": input_data['dialogue_id'][i],
                'ppl': ppls[i],
            }
            # 4 ppls = [12.449083135587186, 12.203526391053927, 11.37369262643693, 11.453226979021665]
            # print(json.dumps(output_data, indent=4))
            output_data_list.append(output_data)

    if world_size > 1:
        result_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(result_list, output_data_list) 
        if local_gpu_rank == 0:
            saved_datas = []
            for result in result_list:
                saved_datas += result
            saved_datas.sort(key=lambda x: x["dialogue_id"]) 
    else:
        saved_datas = output_data_list

    if world_size == 1 or local_gpu_rank == 0:
        result_datas = []
        if len(saved_datas[0]["select_entity"]) == 0:
            dialogue_data = {
                "dialogue_id": saved_datas[0]["dialogue_id"],
                "instruction": saved_datas[0]["instruction"],
                "input": saved_datas[0]["input"],
                "output": saved_datas[0]["output"],
                "entities": saved_datas[0]["entities"],
                "select_entity": [],
                "ppl": saved_datas[0]["ppl"]
            }
        else:
            dialogue_data = {
                "dialogue_id": saved_datas[0]["dialogue_id"],
                "instruction": saved_datas[0]["instruction"],
                "input": saved_datas[0]["input"],
                "output": saved_datas[0]["output"],
                "entities": saved_datas[0]["entities"],
                "select_entity": [{
                    "entity": saved_datas[0]["select_entity"]['text'],
                    "turn": saved_datas[0]["select_entity"]['turn'],
                    "ppl": saved_datas[0]["ppl"]
                }],
                "ppl": 0
            }
        for data in saved_datas[1:]:
            if data["dialogue_id"] == dialogue_data["dialogue_id"]:
                if len(data["select_entity"]['text']) == 0:
                    dialogue_data["ppl"] = data["ppl"]
                else:
                    dialogue_data["select_entity"].append({
                        "entity": data["select_entity"]['text'],
                        "turn": data["select_entity"]['turn'],
                        "ppl": data["ppl"]
                    })
            else:
                # dialogue_data["select_entity"].sort(key=lambda x :x['ppl'], reverse = False)
                dialogue_data["select_entity"].sort(key=lambda x :x['turn'], reverse = False)
                result_datas.append(dialogue_data)
                if len(data["select_entity"]['text']) == 0:
                    dialogue_data = {
                        "dialogue_id": data["dialogue_id"],
                        "instruction": data["instruction"],
                        "input": data["input"],
                        "output": data["output"],
                        "entities": data["entities"],
                        "select_entity": [],
                        "ppl": data["ppl"]
                    }
                else:
                    dialogue_data = {
                        "dialogue_id": data["dialogue_id"],
                        "instruction": data["instruction"],
                        "input": data["input"],
                        "output": data["output"],
                        "entities": data["entities"],
                        "select_entity": [{
                            "entity": data["select_entity"]['text'],
                            "turn": data["select_entity"]['turn'],
                            "ppl": data["ppl"]
                        }],
                        "ppl": 0
                    }
        # dialogue_data["select_entity"].sort(key=lambda x :x['ppl'], reverse = False)
        dialogue_data["select_entity"].sort(key=lambda x: x['turn'], reverse=False)
        result_datas.append(dialogue_data)

        with open(args.output_path, 'w') as f:
            for data in result_datas:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

@torch.inference_mode()
def get_causal(local_gpu_rank, args):
    world_size = args.gpus_num
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_gpu_rank)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    assert tokenizer.eos_token_id == 2, "Tokenizer eos is wrong!!!"
    tokenizer.pad_token_id = 0

    if local_gpu_rank == 0:
        logger = utils.set_file_logger('get_causal_data', os.path.dirname(args.output_path), True)
        logger.info(f'>>>input path: {args.data_path}')
        logger.info(f'>>>output path: {args.output_path}')

    # num_proc = (os.cpu_count()) // world_size
    num_proc = (os.cpu_count())
    datasets = load_dataset('json', data_files=args.data_path)
    if "dialogue_id" in datasets['train'].column_names:
        datasets['train'] = datasets['train'].remove_columns("dialogue_id")
    datasets['train'] = datasets['train'].add_column("dialogue_id", list(range(len(datasets['train']))))
    # datasets = datasets['train'].train_test_split(train_size=0.001, seed=75)  # datasets['train']shuffle，，shuffle
    PROMPT = prompt.persona_prompt(tokenizer, args.cutoff_len, drop=args.drop, use_data_instruction=False)
    if local_gpu_rank == 0:
        import random;
        start = random.randint(1, min(100, len(datasets['train'])-2))
        examples = PROMPT.preprocess_drop_train(datasets['train'][start:start + 1])
        start = random.randint(1, len(examples['input_ids'])-1)
        logger.info(f'>>> prompt example: {tokenizer.decode(examples["input_ids"][start])}')
        logger.info(f'>>> select entity: {examples["select_entity"][start]}')
        logger.info(f'>>> process dataset:')

    # datasets['train'] = Dataset.from_dict(datasets['train'][-5:])
    processed_dataset = datasets['train'].map(PROMPT.preprocess_drop_train, num_proc=num_proc, batched=True, remove_columns = datasets['train'].column_names, batch_size=1)  # ，map， batched=True, remove_columns = datasets['train'].column_names, batch_size=1
    if world_size > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(processed_dataset, shuffle=False)
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=2, sampler=test_sampler)
        test_sampler.set_epoch(0)
    else:
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=2)

    if local_gpu_rank == 0:
        logger.info(f'>>> load model from {args.model_path}')
        logger.info(f'>>> load lora from {args.lora_path}')

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_has_fp16_weight=False,
    )

    if 'no_pos' in args.lora_path:
        if 'lama' in args.model_path:
            from model import MyLlamaForCausalLM
            model = MyLlamaForCausalLM.from_pretrained(
                args.model_path,
                device_map={"": local_gpu_rank},
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_4bit=True,
                use_flash_attention_2=False,
                quantization_config=bnb_config,
            )
        elif 'Qwen' in args.model_path:
            from model import MyQWenLMHeadModel
            model = MyQWenLMHeadModel.from_pretrained(
                args.model_path,
                device_map={"": local_gpu_rank},
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_4bit=True,
                use_flash_attn=False,
                use_flash_attention_2=False,
                quantization_config=bnb_config,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": local_gpu_rank},
            quantization_config=bnb_config
        )

    if len(args.lora_path) > 0:
        # fix the path for local checkpoint
        lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
        if not os.path.exists(lora_bin_path):
            pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
            if os.path.exists(pytorch_bin_path):
                os.rename(pytorch_bin_path, lora_bin_path)
                warnings.warn(
                    "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
                )
            else:
                Exception ('Checkpoint is not Found!')

        model = StreamPeftGenerationMixin.from_pretrained(
            model, args.lora_path, torch_dtype=torch.float16, device_map={"": local_gpu_rank}
        )

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    if local_gpu_rank == 0:
        logger.info(f'>>> caculate ppl:')

    output_data_list = []
    for input_data in tqdm(test_dataset):
        ppls = compute_ppl(input_data, model)
        # ppls = [0 for _ in range(len(input_data['instruction']))]
        for i in range(len(ppls)):
            output_data = {
                'instruction': input_data['instruction'][i],
                'input': input_data['input'][i],
                'output': input_data['output'][i],
                'select_entity': input_data['select_entity'][i],
                "dialogue_id": input_data['dialogue_id'][i],
                'ppl': ppls[i],
            }
            # print(json.dumps(output_data, indent=4))
            output_data_list.append(output_data)
    del model
    if world_size > 1:
        result_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(result_list, output_data_list)
        if local_gpu_rank == 0:
            temp_datas = []
            for result in result_list:
                temp_datas += result
            temp_datas.sort(key=lambda x: x["dialogue_id"])
    else:
        temp_datas = output_data_list

    if local_gpu_rank == 0:
        result_datas = []
        if len(temp_datas[0]["select_entity"]) == 0:
            dialogue_data = {
                "dialogue_id": temp_datas[0]["dialogue_id"],
                "instruction": temp_datas[0]["instruction"],
                "input": temp_datas[0]["input"],
                "output": temp_datas[0]["output"],
                "select_entity": [],
                "ppl": temp_datas[0]["ppl"]
            }
        else:
            dialogue_data = {
                "dialogue_id": temp_datas[0]["dialogue_id"],
                "instruction": temp_datas[0]["instruction"],
                "input": temp_datas[0]["input"],
                "output": temp_datas[0]["output"],
                "select_entity": [{
                    "entity": temp_datas[0]["select_entity"],
                    "ppl": temp_datas[0]["ppl"]
                }],
                "ppl": 0
            }
        for data in temp_datas[1:]:
            if data["dialogue_id"] == dialogue_data["dialogue_id"]:
                if len(data["select_entity"]) == 0:
                    dialogue_data["ppl"] = data["ppl"]
                else:
                    dialogue_data["select_entity"].append({
                        "entity": data["select_entity"],
                        "ppl": data["ppl"]
                    })
            else:
                dialogue_data["select_entity"].sort(key=lambda x :x['ppl'], reverse = False)
                result_datas.append(dialogue_data)
                if len(data["select_entity"]) == 0:
                    dialogue_data = {
                        "dialogue_id": data["dialogue_id"],
                        "instruction": data["instruction"],
                        "input": data["input"],
                        "output": data["output"],
                        "select_entity": [],
                        "ppl": data["ppl"]
                    }
                else:
                    dialogue_data = {
                        "dialogue_id": data["dialogue_id"],
                        "instruction": data["instruction"],
                        "input": data["input"],
                        "output": data["output"],
                        "select_entity": [{
                            "entity": data["select_entity"],
                            "ppl": data["ppl"]
                        }],
                        "ppl": 0
                    }
        dialogue_data["select_entity"].sort(key=lambda x: x['ppl'], reverse=True)
        result_datas.append(dialogue_data)

        with open(args.output_path, 'w') as f:
            for data in result_datas:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        saved_dataset = []
        logger.info(f'>>> caculate ppl:')
        for data in tqdm(result_datas):
            saved_data = data
            if len(saved_data['select_entity']) != len(saved_data['input']):
                select_entity = saved_data['select_entity']
                select_entity.sort(key=lambda x: x['entity'], reverse=True)
                final_select_entity = [select_entity[0]]
                for entity in select_entity[1:]:
                    if entity['entity'] != final_select_entity[-1]['entity']:
                        final_select_entity.append(entity)
                final_select_entity.sort(key=lambda x: x['ppl'], reverse=True)
                saved_data['select_entity'] = final_select_entity
                if len(saved_data['select_entity']) != len(saved_data['input']):
                    logger.warning(f">>> Error !!! The entity number is not equal input number! The dialogue id is : {saved_data['dialogue_id']}")
            for i in range(len(saved_data['select_entity'])):
                ppl = saved_data['select_entity'][i]['ppl']
                entity = saved_data['select_entity'][i]['entity']
                for j, input in enumerate(saved_data['input']):
                    if entity == input['text']:
                        saved_data['input'][j]['ppl'] = ppl
                        break

            try:
                pred_ppls = [data['ppl'] for data in saved_data['input']]
            except Exception as e:
                logger.warning(f">>> Error !!! The input ppl is None! The dialogue id is : {saved_data['dialogue_id']}")
                pred_ppls = [data['ppl'] for data in saved_data['input'] if 'ppl' in data.keys()]
                mean_ppl = np.mean(pred_ppls)
                for i in range(len(saved_data['input'])):
                    if 'ppl' not in saved_data['input'][i].keys():
                        saved_data['input'][i]['ppl'] = mean_ppl
                pred_ppls = [data['ppl'] for data in saved_data['input']]

            # Kmeans
            if np.isnan(np.mean(pred_ppls)):
                continue
            pred_ppls.sort(reverse=True)
            if len(pred_ppls) > 10:
                pred_ppls = pred_ppls[:len(pred_ppls)//2]
                init = np.expand_dims(np.array([pred_ppls[len(pred_ppls) // 2], pred_ppls[0]]), 1)
                y_pred = KMeans(n_clusters=2, init=init, n_init='auto').fit_predict(np.expand_dims(np.array(pred_ppls[:len(pred_ppls) // 2]), 1))
            else:
                init = np.expand_dims(np.array([pred_ppls[len(pred_ppls) // 2], pred_ppls[0]]), 1)
                y_pred = KMeans(n_clusters=2, init=init, n_init='auto').fit_predict(np.expand_dims(np.array(pred_ppls), 1))
            select_num = y_pred.sum()
            max_ppl = pred_ppls[select_num]
            saved_data['select_entity'] = [data for data in saved_data['input'] if data['ppl']>max_ppl]
            saved_dataset.append(saved_data)

        with open(args.output_path, 'w') as f:
            for data in saved_dataset:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

@torch.inference_mode()
def eval_multi_ppl(local_gpu_rank, args):
    world_size = args.gpus_num
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_gpu_rank)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    assert tokenizer.eos_token_id == 2, "Tokenizer eos is wrong!!!"
    tokenizer.pad_token_id = 0

    if local_gpu_rank == 0:
        logger = utils.set_file_logger('get_causal_data', os.path.dirname(args.output_path), True)
        logger.info(f'>>>input path: {args.data_path}')
        logger.info(f'>>>output path: {args.output_path}')
        logger.info(f'>>>model path: {args.model_path}')
        logger.info(f'>>>lora path: {args.lora_path}')

    num_proc = os.cpu_count()
    datasets = load_dataset('json', data_files=args.data_path)
    datasets['train'] = datasets['train'].add_column("dialogue_id", list(range(len(datasets['train']))))
    # datasets = datasets['train'].train_test_split(train_size=0.005, seed=42)
    PROMPT = prompt.persona_prompt(tokenizer, args.cutoff_len, drop=args.drop, use_data_instruction=False)
    if local_gpu_rank == 0:
        import random;
        start = random.randint(1, 100)
        examples = PROMPT.preprocess_multi_drop(datasets['train'][start:start + 1])
        start = random.randint(1, len(examples['input_ids'])-1)
        logger.info(f'>>> prompt example: {tokenizer.decode(examples["input_ids"][start])}')
        logger.info(f'>>> select entity: {examples["select_entity"][start]}')
        logger.info(f'>>> process dataset:')

    processed_dataset = datasets['train'].map(PROMPT.preprocess_multi_drop, num_proc=num_proc, batched=True, remove_columns = datasets['train'].column_names, batch_size=1)  # ，map， batched=True, remove_columns = datasets['train'].column_names, batch_size=1
    if world_size > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(processed_dataset, shuffle=False)
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, sampler=test_sampler)
        test_sampler.set_epoch(0)
    else:
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_has_fp16_weight=False,
    )
    from model import MyLlamaForCausalLM
    if 'no_pos' in args.lora_path:
        model = MyLlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": local_gpu_rank},
            quantization_config=bnb_config
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": local_gpu_rank},
            quantization_config=bnb_config
        )

    if len(args.lora_path) > 0:
        # fix the path for local checkpoint
        lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
        if not os.path.exists(lora_bin_path):
            pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
            if os.path.exists(pytorch_bin_path):
                os.rename(pytorch_bin_path, lora_bin_path)
                warnings.warn(
                    "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
                )
            else:
                assert ('Checkpoint is not Found!')

        model = StreamPeftGenerationMixin.from_pretrained(
            model, args.lora_path, torch_dtype=torch.float16, device_map={"": local_gpu_rank}
        )

    output_data_list = []
    for input_data in tqdm(test_dataset):
        ppls = compute_ppl(input_data, model)
        for i in range(len(ppls)):
            output_data = {
                'instruction': input_data['instruction'][i],
                'input': input_data['input'][i],
                'output': input_data['output'][i],
                'entities': input_data['entities'][i],  # mscESConv，entity。
                'select_entity': {
                    'entity': input_data['select_entity'][i],
                    'ppl': ppls[i]
                },
                "dialogue_id": input_data['dialogue_id'][i],
            }
            # print(json.dumps(output_data, indent=4))
            output_data_list.append(output_data)
    if world_size > 1:
        result_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(result_list, output_data_list)
        if local_gpu_rank == 0:
            saved_datas = []
            for result in result_list:
                saved_datas += result
            saved_datas.sort(key=lambda x: x["dialogue_id"])
    else:
        saved_datas = output_data_list

    if world_size == 1 or local_gpu_rank == 0:
        result_datas = saved_datas
        with open(args.output_path, 'w') as f:
            for data in result_datas:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        result_datas = []
        dialogue_data = {
            "dialogue_id": saved_datas[0]["dialogue_id"],
            "instruction": saved_datas[0]["instruction"],
            "input": saved_datas[0]["input"],
            "output": saved_datas[0]["output"],
            "entities": saved_datas[0]["entities"],
            "select_entity": [
                saved_datas[0]["select_entity"]
            ]
        }
        for data in saved_datas[1:]:
            if data["dialogue_id"] == dialogue_data["dialogue_id"]:
                dialogue_data["select_entity"].append(data["select_entity"])
            else:
                result_datas.append(dialogue_data)
                dialogue_data = {
                    "dialogue_id": data["dialogue_id"],
                    "instruction": data["instruction"],
                    "input": data["input"],
                    "output": data["output"],
                    "entities": data["entities"],
                    "select_entity": [data["select_entity"]],
                }
        result_datas.append(dialogue_data)

        with open(args.output_path, 'w') as f:
            for data in result_datas:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="datasets/CGDIALOG_new/msc.jsonl")
    parser.add_argument('--output_path', type=str, default="output/ppl_CGDIALOG_msc_longchat_7b.jsonl")
    parser.add_argument("--lora_path", type=str, default="outs/msc_longchat_7b")
    parser.add_argument("--model_path", type=str, default="/model/llama-2-7b-chat-hf")
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--eval', type=int, default=0, help="0-;1-caculate ppl;2-eval causal Acc;3-extract causal data;")
    parser.add_argument('--drop', type=int, default=0, help="0-no drop; 1-replace causal")
    args = parser.parse_args()

    if args.eval == 1:
        '''Data Parallel PPL Eval'''
        import random
        MASTER_PORT = random.randint(25000, 30000)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(MASTER_PORT)
        args.gpus_num = torch.cuda.device_count()
        if args.gpus_num > 1:
            mp.spawn(eval_ppl, nprocs=args.gpus_num, args=(args,))
        else:
            eval_ppl(0, args=args)
    elif args.eval == 2:
        print('dataset path isL: ' + args.data_path)
        dataset = open(args.data_path, 'r').readlines()
        dataset = [json.loads(data) for data in dataset]
        metrics = {
            "gold": {1:0, 3: 0, 5: 0},
            "hit_num": {1:0, 3: 0, 5: 0},
            "recall_num": {1:0, 3: 0, 5: 0}
        }
        final_metrics = {
            "gold": 0,
            "hit_num": 0,
            "recall_num": 0
        }
        for data in tqdm(dataset):
            data_ppl = data["ppl"]
            pred_results = data['select_entity']
            max_ppl = max([pred["ppl"] for pred in pred_results])

            pred_results.sort(key=lambda x :x['ppl'], reverse=True)
            pred_entities = [pred["entity"] for pred in pred_results]
            pred_ppls = [pred["ppl"] for pred in pred_results]
            gold_entities = data['entities']
            gold_entities_ids = set()
            for gold_e in gold_entities:
                for i, pred_e in enumerate(pred_entities):
                    if gold_e in pred_e:
                        gold_entities_ids.add(i)
                        break
            gold_entities_ids = list(gold_entities_ids)
            pred_label = [0 for _ in range(len(pred_entities))]
            for i in gold_entities_ids:
                pred_label[i] = 1

            position_turns = []
            for e in pred_entities:
                for i, uttr in enumerate(data['input']):
                    if e == uttr['text']:
                        position_turns.append(i)
            max_turns = max(position_turns)
            position_turns = [pos - max_turns for pos in position_turns]
            if len(position_turns) != len(data['input']):
                pass

            for i in [1, 3, 5]:
                metrics["gold"][i] += sum(pred_label[:i])
                metrics["hit_num"][i] += i
                metrics["recall_num"][i] += len(gold_entities_ids)

            if len(pred_ppls) > 10:
                pred_ppls = pred_ppls[:len(pred_ppls)//2]
                init = np.expand_dims(np.array([pred_ppls[len(pred_ppls) // 2], pred_ppls[0]]), 1)
                y_pred = KMeans(n_clusters=2, init=init, n_init='auto').fit_predict(np.expand_dims(np.array(pred_ppls[:len(pred_ppls) // 2]), 1))
            else:
                init = np.expand_dims(np.array([pred_ppls[len(pred_ppls) // 2], pred_ppls[0]]), 1)
                y_pred = KMeans(n_clusters=2, init=init, n_init='auto').fit_predict(np.expand_dims(np.array(pred_ppls), 1))
            select_num = y_pred.sum()
            final_metrics["gold"] += sum(pred_label[:select_num])
            final_metrics["hit_num"] += select_num
            final_metrics["recall_num"] += len(gold_entities_ids)


        print('--------------------------')
        print("Precision@1: {:.4f}; Precision@3: {:.4f}; Precision@5: {:.4f}".format(metrics["gold"][1]/metrics["hit_num"][1], metrics["gold"][3]/metrics["hit_num"][3], metrics["gold"][5]/metrics["hit_num"][5]))
        print("Recall@1: {:.4f}; Recall@3: {:.4f}; Recall@5: {:.4f}".format(metrics["gold"][1]/metrics["recall_num"][1], metrics["gold"][3]/metrics["recall_num"][3], metrics["gold"][5]/metrics["recall_num"][5]))
        print('final---------------------')
        print("Precision: {:.4f}; Recall: {:.4f}".format(final_metrics["gold"]/final_metrics["hit_num"], final_metrics["gold"]/final_metrics["recall_num"]))
    elif args.eval == 3:
        if not os.path.exists(os.path.dirname(args.output_path)):
            os.mkdir(os.path.dirname(args.output_path))
        output_path = args.output_path
        data_path = args.data_path
        '''Data Parallel PPL Eval'''
        args.gpus_num = torch.cuda.device_count()
        if args.gpus_num > 1:
            import random
            MASTER_PORT = random.randint(25000, 30000)
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(MASTER_PORT)
            mp.spawn(get_causal, nprocs=args.gpus_num, args=(args,))
        else:
            get_causal(0, args=args)