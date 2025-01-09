import json
import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM
import transformers
import argparse
import warnings
import os
import prompt
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from utils import StreamPeftGenerationMixin,StreamLlamaForCausalLM
from vllm import LLM, SamplingParams

'''ddp'''
import torch.distributed as dist
import torch.multiprocessing as mp

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"



def collate_fn(batch):
    prompt, input, output, input_ids, attention_mask, split_tokens = [], [], [], [], [], []
    for data in batch:
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])
        split_tokens.append(data['split_token'])
        prompt.append(data['prompt'])
        input.append(data['input'])
        output.append(data['output'])

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
        model,
        tokenizer,
        split_tokens='Assistant:',
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        min_new_tokens=1,
        repetition_penalty=1.3,
        do_sample=True,
        end_token='</s>',
        **kwargs,
):
    if hasattr(model, 'device'):
        input_ids = input["input_ids"].to(model.device)
        if "attention_mask" in input.keys():
            attention_mask = input["attention_mask"].to(model.device)
        else:
            attention_mask = None
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=max_new_tokens,  # max_length=max_new_tokens+input_sequence
            min_new_tokens=min_new_tokens,  # min_length=min_new_tokens+input_sequence
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        output = generation_output.sequences
        output = tokenizer.batch_decode(output)
        output_str = []
        for i in range(len(output)):
            output_str.append(output[i].split(split_tokens[i])[-1].split(end_token, 1)[0])
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        generation_output = model.generate(
            prompt_token_ids=input["input_ids"].tolist(),
            sampling_params=sampling_params,
            use_tqdm=False
        )
        output_str = []
        for output in generation_output:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            output_str.append(generated_text)
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return output_str

@torch.inference_mode()
def ddp_generate(local_gpu_rank, args):

    world_size = args.gpus_num
    if args.vllm:
        world_size = 1
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_gpu_rank)

    if 'lama' in args.model_path:
         tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True, trust_remote_code=True)
        # assert  tokenizer.eos_token_id == 2, "Tokenizer eos is wrong!!!"
         tokenizer.pad_token_id = 0
    elif 'Qwen' in args.model_path:
         tokenizer = AutoTokenizer.from_pretrained(args.model_path, pad_token='<|endoftext|>', eos_token='<|im_end|>', padding_side='left',add_eos_token=True, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_has_fp16_weight=False,
    )

    if args.vllm:
        model = LLM(model=args.model_path, tensor_parallel_size=args.gpus_num, quantization="AWQ" if 'AWQ' in args.model_path else None, trust_remote_code=True)
    else:
        from model import MyLlamaForCausalLM
        if 'no_pos' in args.lora_path:
            model = MyLlamaForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": local_gpu_rank},
                quantization_config=bnb_config
            )
            model.config.use_cache = False
        else:
            if 'Qwen' in args.model_path:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    use_flash_attn=True,
                    use_flash_attention_2=False,
                    device_map={"": local_gpu_rank},
                    quantization_config=bnb_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    use_flash_attention_2=False,
                    device_map={"": local_gpu_rank},
                    quantization_config=bnb_config
                )

        if len(args.lora_path) > 0:
            model = StreamPeftGenerationMixin.from_pretrained(
                model, args.lora_path, torch_dtype=torch.bfloat16, device_map={"": local_gpu_rank}
            )

        if 'Qwen' in args.model_path:
            for module in model.parameters():
                if hasattr(module, "dtype") and module.dtype == torch.float32:
                    if not isinstance(module, torch.Tensor):
                        module.to(torch.bfloat16)
                    else:
                        module.data = module.data.to(torch.bfloat16)



    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    if args.prompt_type == 'chat':
        PROMPT = prompt.chat_prompt(tokenizer, args.cutoff_len-128, add_eos=False)
    elif args.prompt_type == 'upcr':
        PROMPT = prompt.chat_topic_prompt(tokenizer, args.cutoff_len, add_eos=False)
    elif args.prompt_type == 'persona_chat':
        PROMPT = prompt.persona_prompt(tokenizer, args.cutoff_len, use_data_instruction=True)
    else:
        raise Exception('prompt not support')

    num_proc = (os.cpu_count()) // world_size
    datasets = load_dataset('json', data_files=args.data_path)
    if local_gpu_rank == 0 or world_size == 1:
        import random;start = random.randint(1, 100)
        examples = Dataset.from_dict(datasets['train'][start:start+1]).map(PROMPT.preprocess_gen)
        examples_dataset = DataLoader(examples, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=0)
        result = evaluate_data(list(examples_dataset)[0], model=model, tokenizer=tokenizer, split_tokens=list(examples_dataset)[0]["split_tokens"], max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, end_token=args.end_token)
        for example in examples:
            print(f'>>> prompt example:\n { tokenizer.decode(example["input_ids"]) }')
            print(f'>>> tokenizer example: { example["input_ids"][:10] }...{ example["input_ids"][-10:]}')
        print(f'>>> pred result: {result[0]}')
        print(f">>> gold result: {examples['output'][0]['text']}")

    processed_dataset = datasets['train'].map(PROMPT.preprocess_gen, num_proc=num_proc)
    if world_size > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(processed_dataset, shuffle=False)
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=2, sampler=test_sampler)
        test_sampler.set_epoch(0)
    else:
        test_dataset = DataLoader(processed_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=2)
    output_data_list = []
    for input_data in tqdm(test_dataset):
        result = evaluate_data(input_data, model=model, tokenizer=tokenizer, split_tokens=input_data['split_tokens'], max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, end_token=args.end_token)
        for i in range(len(result)):
            output_data = {
                'prompt': input_data['prompt'][i],
                'input': input_data['input'][i],
                'output': input_data['output'][i],
                # 'entities': input_data['entities'],  # mscESConv，entity。
                'pred_result': result[i],
            }
            # print("pred_result : ", result[i])
            # print("gold_result : ", input_data['output'][i]['text'])
            output_data_list.append(output_data)

    if world_size > 1:
        result_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(result_list, output_data_list)
        if local_gpu_rank == 0:
            output_dataset = []
            for result in result_list:
                output_dataset += result
    else:
        output_dataset = output_data_list

    if local_gpu_rank == 0:
        # if len(args.lora_path) > 0:
        #     args.output_path = args.output_path + '_' + args.model_path.split('/')[-1] + '_' + args.lora_path.split('/')[-1] + '.jsonl'
        # else:
        #     args.output_path = args.output_path + '_' + args.model_path.split('/')[-1] + '.jsonl'
        # if not os.path.exists(args.output_path.split('/', 1)[0]):
        #     os.mkdir(args.output_path.split('/', 1)[0])
        # print('Output data path is :', args.output_path)
        with open(args.output_path, 'w') as f:
            for data in output_dataset:
                f.write(json.dumps(data, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/model/13B_hf")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="datasets/CGDIALOG/ESConv.jsonl")
    parser.add_argument("--output_path", type=str, default="output/ESConv")
    parser.add_argument("--prompt_type", type=str, default="chat")
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--num_beams", type=int, default=1, help='beam_search depth')
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_local", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--kind", type=str, default="raw", choices=['raw', 'rsm_summary', 'rsm_response', 'constrain'])
    parser.add_argument("--vllm", action='store_true')
    args = parser.parse_args()

    if 'lama' in args.model_path:
        args.end_token = '</s>'
    elif 'Qwen' in args.model_path:
        args.end_token = '<|im_end|>'
    print(args)

    '''Data Parallel Generation'''
    import random
    MASTER_PORT = random.randint(25000, 30000)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(MASTER_PORT)
    args.gpus_num = torch.cuda.device_count()
    if args.kind == 'raw':
        generate_func = ddp_generate
    if args.gpus_num > 1 and (not args.vllm):
        mp.spawn(generate_func, nprocs=args.gpus_num, args=(args, ))
    else:
        generate_func(0, args=args)
