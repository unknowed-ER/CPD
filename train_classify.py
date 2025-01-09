from transformers import (
    TrainerCallback,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoModelWithLMHead,
    get_polynomial_decay_schedule_with_warmup, 
    GenerationConfig,
)
import tqdm
from transformers.trainer_callback import PrinterCallback
import os
import sys
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
import transformers
import random
from functools import partial
import utils
import prompt
import middleware
import wandb
import pandas as pd

def focal_forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    from transformers import SequenceClassifierOutput
    from torchvision.ops import focal_loss
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
        # if self.num_labels == 1:
        #     loss_fct = MSELoss()
        #     loss = loss_fct(logits.view(-1), labels.view(-1))
        # else:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss_fct = focal_loss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids).long()
    # NOTE here can't be float16, must be float:
    # class 'RuntimeError'> "softmax_lastdim_kernel_impl" not implemented for 'Half'  
    preds = torch.softmax(torch.tensor(pred.predictions,dtype=float),dim=-1)
    # out[i][j] = preds[i][labels[i][j]]
    probs = torch.gather(preds, 1,labels.view(-1, 1))
    acc = torch.mean(probs).item()
    return {
        'accuracy': round(acc,6)
    }

def load_data(data_path):
    data = utils.from_jsonl( data_path )
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    start = random.randint(0, len(dataset)-1)
    examples = Dataset.from_dict(dataset[start:start+1])
    return dataset, examples

def train_classifier(
    args,
):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    args.gradient_accumulation_steps = args.total_batch // args.micro_batch
    if ddp:
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
    logger = utils.set_file_logger('transformers.trainer', args.output_path, use_console=True)
    args.ip = utils.extract_ip()
    args.time= utils.get_time()
    logger.warning(f'>>> output in  {args.output_path}')
    logger.info(f'>>> using {args}')
    
    # 0. prepare tokenizer, model
    current_device = utils.get_local_rank2()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
    ).to(current_device)
    preprocess = lambda example: tokenizer(
        example['text'], 
        truncation=True, 
        padding=False, 
        max_length=128,
    )

    # 1. load data
    train_data, examples = load_data(args.data_path) 
    eval_data, eval_examples = load_data(args.dev_data_path) 
    examples = examples.map(preprocess)
    for example in examples:
        label = example["labels"] if 'labels' in example.keys() else example["label"]
        logger.info(f'>>> example:\n { tokenizer.decode(example["input_ids"]) }')
        logger.info(f'>>> tokenizer example: { example["input_ids"][:10] }...{ example["input_ids"][-10:]}')
        logger.info(f'>>> tokenizer labels: { label }')
    train_data = train_data.shuffle().map(preprocess, num_proc=os.cpu_count())
    eval_data = eval_data.shuffle().map(preprocess, num_proc=os.cpu_count())

    # 2. prepare trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.total_batch//args.micro_batch,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epoch,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_strategy="epoch",
            logging_first_step=True, # convenient
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2, # save best / last
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            output_dir=args.output_path,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.use_wandb else [],
        ),
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    if not args.use_wandb:
        wandb.init(mode='disabled')

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # save args
    utils.to_json(args.__dict__, f'{args.output_path}/train_args.json')

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(f'{args.output_path}/best')
    if args.use_wandb:
        wandb.config.update(args, allow_val_change=True)
        logger.info({'wandb-url': wandb.run.get_url()})

def _start(
    *, 
    data_path: str,
    dev_data_path: str,
    num_labels: int,
    output_path: str,
    model_path:str='bert-base-uncased',
    cutoff: int=512,
    use_wandb:bool=False, 
    micro_batch:int=4,
    total_batch:int=32,
    warmup_ratio:float= 0.05,
    num_epoch:int=100,
    latent_size:int=64,
    learning_rate:float=5e-5,
    log_steps:int=4,
    int8:bool=False,
    resume_from_checkpoint: bool=False,
):
    import inspect
    frame = inspect.currentframe()
    names, _, _, locals = inspect.getargvalues(frame)
    args = utils.HyperParams().from_inspect(names, locals)
    train_classifier(args)
    # setattr(transformers.BertForSequenceClassification,'forward',focal_forward)

if __name__ == "__main__":
    import defopt
    try:
        defopt.run(_start)
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)