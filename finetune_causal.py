from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import pathlib
import json
# from accelerate import Accelerator
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NLLLoss
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainerCallback, GenerationConfig, BitsAndBytesConfig, AutoModel, AutoTokenizer
import os
import sys
import logging
from peft.tuners.lora import Linear8bitLt
import torch.nn.functional as F
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset

import transformers
import argparse
import warnings
from tqdm import tqdm
from functools import partial
import utils
import prompt

DEBUG = os.environ.get('DEBUG', False)


class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self.epoch = 0
        self.trainer = trainer
        self.logger = logging.getLogger("transformers.trainer")
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        # if (self.epoch + 1) % 3 == 0:
        #     self.trainer._save_checkpoint(self.trainer.model, trial=None, metrics=None)
        # self.epoch += 1
        pass

    # save model
    def on_save(self, args, state, control, model, **kwargs):
        # self.test(model, args,state)
        # 
        # checkpoint_folder = f"checkpoint-{state.global_step}"
        # run_dir = self.trainer._get_output_path(trial=None)
        # output_path = os.path.join(run_dir, checkpoint_folder)
        # json.dump(open(f'{output_path}/adapter_config.json'))
        pass

    def test(self, model, args, state):
        pass

    def on_log(self, args, state, control, logs, **kwargs):
        # if args.debug_nan:
        #     logs.update({'outliner': utils.max_outliner})
        self.logger.info(logs)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        self.logger.info(f'>>> preprocess dataset before epoch:')
        train_data = self.trainer.dataset_raw.map(self.trainer.PROMPT.causal_train, num_proc=self.trainer.num_proc, batched=True, remove_columns = self.trainer.dataset_raw.column_names, batch_size=1)
        # train_data = train_data.remove_columns(['instruction', 'input', 'output'])
        self.trainer.train_dataset._data = train_data._data  # dataloader，。。
        self.logger.info(f'>>> preprocess dataset finish; Raw dataset len is : {len(self.trainer.dataset_raw)}. Processed dataset is : {len(train_data)}')

def compute_loss(self, model, inputs, return_outputs=False):
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    ite_label = inputs['labels'][0, 0].tolist() == -99
    inputs['labels'][inputs['labels']==-99] = -100
    outputs = model(**inputs)
    if DEBUG:
        idxs = torch.where(inputs['labels'][0]!=-100)[0]
        in_text = self.tokenizer.decode(inputs['labels'][0][idxs])
        out_text = self.tokenizer.decode(torch.argmax(outputs['logits'],dim=-1)[0][idxs])
        self.logger.info(f'>>>in: {in_text}')
        self.logger.info(f'>>>out: {out_text}')
    if torch.isnan(outputs.loss):
        print('loss is nan')
        raise Exception('loss is nan')
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        loss = self.label_smoother(outputs, labels, shift_labels=True)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        if ite_label:
            # Shift so that tokens < n predict n
            logits = outputs.logits
            labels = inputs['labels']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.softmax(dim=-1)
            shift_logits = 1 - shift_logits  # 
            shift_logits = shift_logits.log()
            # Flatten the tokens
            loss_fct = NLLLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss

# for debug:
def lora_int8_forward(self, x: torch.Tensor):
    
    # super()
    result = super(Linear8bitLt, self).forward(x)

    if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
        return result
    elif self.r[self.active_adapter] > 0:
        if not torch.is_autocast_enabled():
            expected_dtype = result.dtype

            if x.dtype != torch.float32:
                x = x.float()
            output = (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                ).to(expected_dtype)
                * self.scaling[self.active_adapter]
            )
        else:
            output = (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        result += output
    return result

def lora_fp16_forward(self, x ):

    def transpose(weight, fan_in_fan_out):
        return weight.T if fan_in_fan_out else weight
    
    previous_dtype = x.dtype
    if self.active_adapter not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter].weight.dtype)

        result += (
            self.lora_B[self.active_adapter](
                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
            )
            * self.scaling[self.active_adapter]
        )
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)
    return result

def debug_patcher(model, args):
    from types import MethodType
    # replace instance-based methods
    _mlp = utils.get_module_by_name(model, 'model.model.layers.30.mlp.down_proj')
    _mlp.forward = MethodType(lora_int8_forward if args.int8 else lora_fp16_forward,_mlp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--model_path", type=str, default='/model/Llama-2-7b-hf')
    parser.add_argument("--num_epoch", type=int, default=3)
    parser.add_argument("--micro_batch", type=int, default=1)
    parser.add_argument("--save_strategy", type=str)
    parser.add_argument("--dtype", type=str, default='fp16')
    parser.add_argument("--total_batch", type=int, default=128)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--test_size", type=int, default=0)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--debug_nan", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--use_data_instruction", action='store_true')  # personaprompt
    parser.add_argument("--int8", action='store_false')
    parser.add_argument("--lora", action='store_false')
    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "disable"

    args.gradient_accumulation_steps = args.total_batch // args.micro_batch

    # accelerator = Accelerator()
    # device_index = accelerator.process_index
    # device_map = {"": device_index}
    device_map = {"": 0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size


    logger = utils.set_file_logger('transformers.trainer', args.output_path, True)
    logger.info(f'>>> world size: {world_size}')
    logger.info(f'>>> device map: {device_map}')
    logger.info(f'>>> processing data from {args.data_path}')
    logger.info(f'>>> using {args}')

    train_tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    # assert train_tokenizer.eos_token_id == 2, "Tokenizer eos is wrong!!!"
    train_tokenizer.pad_token_id = 0
    if args.prompt_type == 'instruct':
        PROMPT = prompt.instruct_prompt(train_tokenizer, args.cutoff_len)
    elif args.prompt_type == 'chat':
        PROMPT = prompt.chat_prompt(train_tokenizer, args.cutoff_len)
    elif args.prompt_type == 'attribute_autoencoder':
        PROMPT = prompt.attribute_prompt(train_tokenizer, args.cutoff_len)
    elif args.prompt_type == 'attribute_autoencoder_few':
        PROMPT = prompt.attribute_fewshot_prompt(train_tokenizer, args.cutoff_len)
    elif args.prompt_type == 'attribute_context':
        assert args.cutoff_len > 512
        PROMPT = prompt.attribute_context(train_tokenizer, args.cutoff_len)
    elif args.prompt_type == 'upcr':
        PROMPT = prompt.chat_topic_prompt(train_tokenizer, args.cutoff_len)
    elif args.prompt_type == 'persona_chat':
        PROMPT = prompt.persona_prompt(train_tokenizer, args.cutoff_len, use_data_instruction=args.use_data_instruction)
    else:
        raise Exception('not support')
    # check tokenizer
    # data = load_dataset('json', data_files=json.loads(args.data_path.replace("\'", "\"")))

    data = load_dataset('json', data_files=args.data_path)
    import random;

    start = random.randint(1, min(100, len(data['train'])-1))
    num_proc = os.cpu_count()
    logger.info(f'>>> statistic dataset:')
    data['train'] = data["train"].map(PROMPT.get_entity_turns, num_proc=num_proc)
    PROMPT.statistic(data['train'])
    logger.info(f'>>> statistic dataset finish !')
    examples = Dataset.from_dict(data['train'][start:start + 1]).map(PROMPT.causal_train, batched=True, remove_columns = data['train'].column_names, batch_size=1)
    # for example in examples:
    example = examples[0]
    logger.info(f'>>> prompt example:\n {train_tokenizer.decode(example["input_ids"])}')
    logger.info(f'>>> tokenizer labels: {train_tokenizer.decode([0 if l == -100 else l for l in example["labels"]])}')
    logger.info(f'>>> tokenizer example: {example["input_ids"][:10]}...{example["input_ids"][-10:]}')

    logger.info(f'>>> preprocess dataset :')
    # # 3. speedup dataset processing by multi-process
    # train_data = data["train"].map(PROMPT.causal_train, num_proc=num_proc)
    train_data = data["train"].map(PROMPT.causal_train, num_proc=num_proc, batched=True, remove_columns=data["train"].column_names, batch_size=1)
    logger.info(f'>>> preprocess dataset finish. Raw dataset len is : {len(data["train"])}. Processed dataset is : {len(train_data)}')

    # 2. load model and checkpoints
    logger.info(f'>>> load model from {args.model_path}')

    if "no_pos" in args.output_path:
        from model import MyLlamaForCausalLM
        LoadModel = MyLlamaForCausalLM
    else:
        LoadModel = LlamaForCausalLM
    model = LoadModel.from_pretrained(
        args.model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float16,
        # use_flash_attention_2=False,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
        ),
    )
    if args.int8:
        model = prepare_model_for_kbit_training(model)
    elif not args.lora:
        utils.freeze(model)

    if args.debug_nan:
        utils.debug_nan(model)
    if args.lora:
        config = LoraConfig(
            r=64,
            lora_alpha=8,
            target_modules=utils.find_all_linear_names(model),
            lora_dropout=0.05,
            bias="none",  # llama linear has no bias
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # model.config.torch_dtype = torch.float32
        if args.resume:
            lora_bin_path = os.path.join(args.output_path, "adapter_model.bin")
            logger.info({'load lora weight from': lora_bin_path})
            if not os.path.exists(lora_bin_path):
                raise Exception('Checkpoint is not Found!')
            adapters_weights = torch.load(lora_bin_path)
            set_peft_model_state_dict(model, adapters_weights)


    # logger.info({'trainable_parameter': utils.get_trainable_name(model)})
    # logger.info({'trainable_dtype': utils.get_dtype_name(model)})
    logger.info({'trainable_size': utils.get_trainable_numel(model)})
    # logger.info({'memory': str(utils.get_transformers_memory(model)) + 'GB'})
    # utils.verify_model_dtype(model)

    # replace class-based methods
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        tokenizer=train_tokenizer,
        args=transformers.TrainingArguments(
            dataloader_num_workers=4,
            remove_unused_columns=False,
            per_device_train_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epoch,
            learning_rate=3e-4,
            bf16=args.dtype == 'bf16',
            fp16=args.dtype == 'fp16',
            logging_steps=args.log_steps,
            logging_first_step=True,  # convenient
            evaluation_strategy="no",
            save_strategy=args.save_strategy,
            eval_steps=None,
            save_steps=args.save_steps,
            output_dir=args.output_path,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.wandb else [],
            optim="paged_adamw_8bit",
        ),
        data_collator=PROMPT.data_collator()
    )
    setattr(transformers.Trainer, 'compute_loss', compute_loss)
    setattr(transformers.Trainer, 'logger', logger)
    setattr(transformers.Trainer, 'num_proc', num_proc)
    setattr(transformers.Trainer, 'PROMPT', PROMPT)
    setattr(transformers.Trainer, 'dataset_raw', data['train'])
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.add_callback(CustomCallback(trainer))
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if list(pathlib.Path(args.output_path).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    trainer.save_state()
    model.save_pretrained(args.output_path)

