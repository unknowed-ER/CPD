import logging
from logging.handlers import RotatingFileHandler
import multiprocessing, threading, logging, sys, traceback
from accelerate import Accelerator
import sys
import os
import torch
import json
from typing import Optional, Tuple, Union, List, Callable
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.beam_search import BeamSearchScorer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    GenerationMixin,
)
import functools
import warnings
import peft
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import torch.distributed as dist
from torch import nn
import copy
from datetime import datetime
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from accelerate import dispatch_model, infer_auto_device_map
from peft.utils import PeftType, set_peft_model_state_dict
import traceback
import socket
import pandas as pd
from platform import uname
import numpy as np
import re
import yaml
import random
import sys
import pdb
import subprocess
import numpy as np

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def post_mmortem(t=None):
    # handling the default
    if t is None:
        # sys.exc_info() returns (type, value, traceback) if an exception is
        # being handled, otherwise it returns None
        t = sys.exc_info()[2]
    if t is None:
        raise ValueError("A valid traceback must be passed if no "
                         "exception is being handled")
    p = ForkedPdb()
    p.reset()
    p.interaction(None, t)

pdb.set_mtrace = ForkedPdb().set_trace
# pdb.set_trace=lambda:0
#  pdb.set_trace=pdb.set_ttrace
pdb.set_ttrace = pdb.set_trace
pdb.post_mmortem = post_mmortem

DEBUG=os.environ.get('DEBUG',False)

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def print_rank_last(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            print(message, flush=True)
    else:
        print(message, flush=True)

def bash(cmd:str):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def in_wsl() -> bool:
    return 'microsoft-standard' in uname().release

def mean(data):
    if isinstance(data, dict):
        return np.array(list(data.values())).mean()
    elif isinstance(data, list):
        return np.array(data).mean()

def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('8.8.8.8', 80))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP

def word_cnt(sent):
    return len(sent.split())

def mean(data, weight=None):
    # {a:x,b:y,c:z} -> (x+y+z) /3
    if isinstance(data, dict):
        return np.average(np.array(list(data.values())),weights=weight)
    elif isinstance(data, list):
        return np.average(np.array(data),weights=weight)
    else:
        raise Exception('no implementation')

def dict_mean(dict_list):

    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_transformers_memory(model,unit='gb' ):
    # NOTE only for huggingface transformers
    divisor = 1
    if unit.lower() == 'kb':
        divisor = 1024
    elif unit.lower() == 'mb':
        divisor = 1024*1024
    elif unit.lower() == 'gb':
        divisor = 1024*1024*1024
    else:
        raise ValueError()
    return model.get_memory_footprint() / divisor

def get_memory(info_name='memory_allocated', unit='G'):
    if info_name == 'memory_allocated':
        current_value = torch.cuda.memory.memory_allocated()
    elif info_name == 'max_memory_allocated':
        current_value = torch.cuda.memory.max_memory_allocated()
    elif info_name == 'memory_reserved':
        current_value = torch.cuda.memory.memory_reserved()
    elif info_name == 'max_memory_reserved':
        current_value = torch.cuda.memory.max_memory_reserved()
    else:
        raise ValueError()

    divisor = 1
    if unit== 'K':
        divisor = 1024
    elif unit == 'M':
        divisor = 1024*1024
    elif unit == 'G':
        divisor = 1024*1024*1024
    else:
        raise ValueError()

    return current_value/divisor #, diff_value/divisor

def reimport(nameOfModule):
    # reimport a module while interactive
    import importlib
    importlib.reload(nameOfModule)

def unwrap_torch_compile(state_dict):
    # orderdict，cmp
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k.replace('_orig_mod.','')] = state_dict[k]
    del state_dict
    return new_state_dict

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

accelerator =None
def get_local_rank2():
    # return accelerator.local_process_index
    global accelerator
    if accelerator is None:
        accelerator = Accelerator()
    return accelerator.local_process_index
    # return 'cpu'
    # return 'cpu' # for debug

def gpu_clean():
    import gc;
    gc.collect()
    torch.cuda.empty_cache()

def argsmax(array):
    return max(enumerate(array), key=lambda x: x[1])

def remove_chinese_space(text):
    text = re.sub(r'(?<=[^a-zA-Z0-9]) +(?=[^a-zA-Z0-9])', '', text)
    text = re.sub(r'(?<=[a-zA-Z0-9+]) +(?=[^a-zA-Z0-9])', '', text) # Ka+ Color Edition
    text = re.sub(r'(?<=[^a-zA-Z0-9+]) +(?=[a-zA-Z0-9])', '', text)
    return text

def get_tokenizer(path, max_length):
    if 'llama' in path:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            path,
            model_max_length=max_length,
            padding_side="right",
            add_eos_token=True
        )
        tokenizer.pad_token_id = 0
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            model_max_length=max_length,
            padding_side="right",
            add_eos_token=True
        ) 
    return tokenizer

# 
MODEL_DIR= os.environ.get('MODEL_DIR', 'model')
RANK=int(os.environ.get('LOCAL_RANK', 0))

def download_hf(path, outpath, torch_dtype, type='causal'):
    # causal raw seq2seqlm classification
    from transformers import AutoModel,AutoModelForCausalLM,AutoModelForSeq2SeqLM,AutoTokenizer
    if type == 'causal':
        AutoModelForCausalLM.from_pretrained(path,use_auth_token=True, torch_dtype=torch_dtype).save_pretrained(outpath)
    elif type == 'seq2seq':
        AutoModelForSeq2SeqLM.from_pretrained(path,use_auth_token=True, torch_dtype=torch_dtype).save_pretrained(outpath)
    elif type=='classification':
        # AutoModelForSequenceClassification
        # AutoModelForTokenClassification
        # AutoModelForQuestionAnswering
        pass
    elif type =='raw':
        AutoModel.from_pretrained(path,use_auth_token=True, torch_dtype=torch_dtype).save_pretrained(outpath)
    else:
        raise NotImplementedError
    AutoTokenizer.from_pretrained(path, use_auth_token=True).save_pretrained(outpath)
    import gc;gc.collect()


class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)

class HyperParams:
    def from_inspect(self, args, locals):
        for n in args:
            setattr(self, n, locals[n])
        return self

    def from_dict(self, dicts):
        for n,v in dicts.items():
            setattr(self, n, v)
        return self

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}

        for name, value in sorted(self.__dict__.items()):
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))



def extract_caller():
    import traceback
    stack = traceback.format_list(traceback.extract_stack()[::-1])
    for s in stack:
        if 'modeling_llama' in s:
            return s.split('\n')[-2]
    return ''

def printf(*args,**kargs):
    if os.environ.get('DEBUG',False):
        end = '\n'
        if 'end' in kargs:
            end = kargs['end']
        print(*args, end=end, flush=True)


from contextlib import contextmanager

@contextmanager
def evaluating(model):
    state = model.training    
    try:
        model.eval()
        logging.getLogger('transformers.trainer').setLevel(logging.WARNING)
        yield model
    finally:
        if state:
            model.train()
        logging.getLogger('transformers.trainer').setLevel(logging.INFO)

def disable_dropout(model):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            model.eval()
            func(*args, **kwargs)
            model.train()
        return wrapper
    return decorator

def disable_dropout2(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args[0].eval()
        func(*args, **kwargs)
        args[0].train()
    return wrapper

class ColorFormatter(logging.Formatter):

    grey = "\x1b[30;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    purple = "\x1b[35;20m"
    light_blue = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__(fmt)
        self.FORMATS = {
            logging.DEBUG: self.grey + fmt + self.reset,
            logging.INFO: self.blue + fmt + self.reset,
            logging.WARNING: self.yellow + fmt + self.reset,
            logging.ERROR: self.red + fmt + self.reset,
            logging.CRITICAL: self.bold_red + fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# https://github.com/jruere/multiprocessing-logging
def set_console_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler(sys.stdout)
    if DEBUG:
        consoleHandler.setLevel(logging.DEBUG if RANK in [-1,0] else logging.WARNING)
    else:
        consoleHandler.setLevel(logging.INFO if RANK in [-1,0] else logging.WARNING)
    consoleHandler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s %(message)s"))
    logger.addHandler(consoleHandler)
    return logger

def set_file_logger(name, dir, use_console=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    os.makedirs(dir, exist_ok=True)

    if use_console:
        logger.propagate = False # disable default handler
        consoleHandler = logging.StreamHandler(sys.stdout)
        if DEBUG:
            consoleHandler.setLevel(logging.DEBUG if RANK in [-1,0] else logging.WARNING )
        else:
            consoleHandler.setLevel(logging.INFO if RANK in [-1,0] else logging.WARNING )
        consoleHandler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s %(message)s"))
        logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(os.path.join(dir,'session.log'), mode='a') 
    fileHandler.setLevel(logging.INFO if RANK in [-1,0] else logging.WARNING)
    fileHandler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s %(message)s"))
    logger.addHandler(fileHandler)
    return logger

def to_jsonl(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode) as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')

def from_jsonc(path):
    # support for json with comment 
    import jstyleson
    return jstyleson.load(open(path))

def from_json(path):
    return json.load(open(path))

def from_jsonl(path):
    return [json.loads(line) for line in open(path, 'r',encoding='utf8') ]

def to_yaml(data,path):
    yaml.safe_dump(data,open(path,'w'))

def from_yaml(path):
    return yaml.safe_load(open(path))

def to_json(data, path, mode='w'):
    if mode == 'a' and os.path.exists(path):
        old_data = from_json(path)
        data = old_data + data
    json.dump(data, open(path, 'w', encoding='utf8'), ensure_ascii=False)

# next(iter(data.items()))[1].keys()
def to_excel(data, path, index=None, columns=None, mode='w'):

    if columns is None:
        # text_df(index, 'b')
        # NOTE : { 'a':{'x''y'},'b':{'x''y'}} => rows: x,y columns: a,b
        df = pd.DataFrame(data,index=index).T
        if mode == 'a':
            if os.path.exists(path):
                previous = pd.read_excel(path,index_col=0)
                df = pd.concat([previous,df])
                df.to_excel(path,index=True)
                return
        df.to_excel(path,index=True)
    # given column
    elif index is None:
        df = pd.DataFrame(data,columns = columns)

    df.to_excel(path,index=False)

def from_excel(path):
    df = pd.read_excel(path).to_dict('records')
    return df

def pyspark2jsonl(inputpath,outputpath):
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SQLContext
    # example: pyspark2jsonl(inputpath='/mnt/c/Users/lzy/Downloads/train-00000-of-00001-e7b340ef75ddca08.parquet', outputpath='zhihu_kol.jsonl')
    sc = SparkContext(appName="Transform Pq to Csv")
    sqlContext = SQLContext(sc)
    df = sqlContext.read.parquet(inputpath)
    results = df.toJSON().map(lambda j: json.loads(j)).collect()
    utils.to_jsonl(results, outputpath)

def get_trainable_name(model):
    return [n for n,p in model.named_parameters() if p.requires_grad]

def get_dtype_name(model):
    return [ (n,p.dtype) for n,p in model.named_parameters() ]

def verify_model_dtype(model):
    from collections import defaultdict
    dtype2param_num = defaultdict(int)  # 
    dtype2param_name = defaultdict(list)  # 
    dtype2trainable_param_num = defaultdict(int)  # 
    dtype2trainable_param_name = defaultdict(list)  # 
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # ，
    total = 0
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # ，
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)

def get_nograd_name(model):
    return [n for n, p in model.named_parameters() if not p.requires_grad]

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

max_outliner = 0
import warnings
def debug_nan(model):

    class nan_hook:
        def __init__(self,name, module):
            # module.__class__.__name__ 
            self.name=name
            module.register_forward_hook(self._hook)

        def _hook(self, module, inp, output):
            # lora
            # printf(self.name)
            
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):

                if out is None:
                    continue
                if self.name == 'model':
                    # dataclass
                    continue
                if isinstance(out, dict):
                    # for k,v in out.__dict__.items():
                    #     try:
                    #         print(k, v.max())
                    #     except:
                    #         pass
                    return
                # else:
                #     printf(out.max())
                
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    raise RuntimeError(f"Found NAN in {self.name} output {i} at indices: ", nan_mask.nonzero())
                inf_mask = torch.isinf(out)
                if inf_mask.any():
                    raise RuntimeError(f"Found INF in {self.name} output {i} at indices: ", inf_mask.nonzero())
                outliner = out.abs().max()
                if outliner > 1000:
                    # raise RuntimeError(f"Found outlier in {self.name} output {out_max}: ", out.argmax())
                    # warnings.warn(f"Found outlier in {self.name} output {out_max}: {out.argmax()}" )
                    global max_outliner
                    max_outliner = max(max_outliner, outliner.item())

            # torch.isinf(hidden_states).any()
            # torch.isinf(hidden_states).nonzero()
    
    # for submodule in model.modules():
    for name,submodule in model.named_modules():
        nan_hook(name, submodule)
# NaN gradients are expected occasionally, and scaler.step(optimizer) should safely skip the step.
# NaN loss is not expected, and indicates the model is probably corrupted.
# with autocast(enabled=False)

def get_module_by_name(module, access_string):
    # get_module_by_name(resnet34, 'layer1.0.relu')
    names = access_string.split(sep='.')
    return functools.reduce(getattr, names, module)


def freeze(model):
    model.requires_grad_(False)

def get_trainable_numel(model, unit='b'):
    divisor = 1
    if unit== 'k':
        divisor = int(1e3)
    elif unit == 'm':
        divisor = int(1e6)
    elif unit == 'b': # billion=10yi
        divisor = int(1e9)
    else:
        raise ValueError()

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += param.numel()
    return {
        "trainable params": trainable_params/divisor,
        "all params": all_param/divisor, 
        "trainable%": 100 * trainable_params / all_param,
    }

def get_grads(model):
    named_grads ={}
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_grads[name] = param.grad
    return named_grads

def get_parameter_updates(optimizer):
    pass

class StreamGenerationMixin(GenerationMixin):
    # support for streamly generation
    # TODO: group_beam_search
    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        **kwargs,
    ):
        if is_deepspeed_zero3_enabled() and dist.world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

        if kwargs.get("attention_mask", None) is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(
                kwargs["input_ids"].shape[0], self.peft_config.num_virtual_tokens
            ).to(kwargs["input_ids"].device)
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, kwargs["attention_mask"]), dim=1
            )
        if kwargs.get("position_ids", None) is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
            )
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
            )
            kwargs["token_type_ids"] = None

        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        bos_token_id, eos_token_id, pad_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
            generation_config.pad_token_id,
        )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_seq_length
            )
        if generation_config.min_new_tokens is not None:
            generation_config.min_length = (
                generation_config.min_new_tokens + input_ids_seq_length
            )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = (
                "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            )

        # 2. Set generation parameters if not already defined
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        # beam=1 and do_sample=True
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )
        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        if is_greedy_gen_mode:
            # 11. run greedy search
            return self.stream_greedy_search(
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                synced_gpus,
                **model_kwargs,
            )
        elif is_sample_gen_mode:
            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            return self.stream_sample(
                generation_config,
                input_ids,
                logits_processor,
                logits_warper,
                stopping_criteria,
                synced_gpus,
                **model_kwargs,
            )
        elif is_beam_gen_mode:
            return self.stream_beam_search(
                generation_config,
                input_ids,
                logits_processor,
                stopping_criteria,
                synced_gpus,
                **model_kwargs,
            )
        elif is_beam_sample_gen_mode:
            # interleave input_ids with `num_beams` additional sequences per batch
            return self.stream_beam_sample(
                input_ids,
                logits_processor,
                logits_warper,
                stopping_criteria,
                generation_config,
                synced_gpus,
                **model_kwargs,
            )
        else:
            raise Exception('not implement')
        
    def stream_sample(
        self,
        generation_config,
        input_ids,
        logits_processor,
        logits_warper,
        stopping_criteria,
        synced_gpus,
        **model_kwargs,
    ):
        bos_token_id, eos_token_id, pad_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
            generation_config.pad_token_id,
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False  # used by synced_gpus only
        scores=()
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            next_token_logits = outputs.logits[:, -1, :]
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            yield input_ids
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
            
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        yield input_ids

    def stream_beam_sample(
        self,
        input_ids,
        logits_processor,
        logits_warper,
        stopping_criteria,
        generation_config,
        synced_gpus,
        **model_kwargs,
    ):
        bos_token_id, eos_token_id, pad_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
            generation_config.pad_token_id,
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        num_beams = generation_config.num_beams
        batch_size, cur_len = input_ids.shape[0], input_ids.shape[-1]
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=input_ids.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams * generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        scores = ()
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            # Note: logits warpers are intentionally applied after adding running beam scores. On some logits warpers
            # (like top_p) this is indiferent, but on others (like temperature) it is not. For reference, see
            # https://github.com/huggingface/transformers/pull/5420#discussion_r449779867
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=None,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            yield input_ids
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=None,
        )
        yield sequence_outputs["sequences"]

    def stream_greedy_search(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        **model_kwargs,
    ):
        # init values
        bos_token_id, eos_token_id, pad_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
            generation_config.pad_token_id,
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        # init attention / hidden states / scores tuples
        scores = () 
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            yield input_ids
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True
        yield input_ids

    def stream_beam_search(
        self,
        generation_config,
        input_ids,
        logits_processor,
        stopping_criteria,
        synced_gpus,
        **model_kwargs,
    ):

        # 10. go into beam search generation modes
        # 11. prepare beam search scorer
        bos_token_id, eos_token_id, pad_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
            generation_config.pad_token_id,
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        num_beams = generation_config.num_beams
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=input_ids.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # beam_search logits
        batch_beam_size, cur_len = input_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) hack: adjust tokens for Marian.
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[
                :, None
            ].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=None,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            # increase cur_len
            cur_len = cur_len + 1

            yield input_ids

            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        final_result = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=None,
        )
        yield final_result["sequences"]

class StreamLlamaForCausalLM(LlamaForCausalLM, StreamGenerationMixin):
    pass

class StreamPeftGenerationMixin(PeftModelForCausalLM, StreamGenerationMixin):
    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name="default", is_trainable=False,  **kwargs):
        # work in peft==0.3.0
        if peft.__version__ >= '0.3.0' and peft.__version__ != '0.3.0.dev0':
            # load the config
            from peft import PromptLearningConfig
            config = LoraConfig.from_pretrained(model_id)

            if (getattr(model, "hf_device_map", None) is not None) and len(
                set(model.hf_device_map.values()).intersection({"cpu", "disk"})
            ) > 0:
                remove_hook_from_submodules(model)

            if isinstance(config, PromptLearningConfig) and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                config.inference_mode = not is_trainable

            # here is the hack
            model = cls(model, config, adapter_name)
            model.load_adapter(model_id, adapter_name, **kwargs)
            # NOTICE
            model.base_model_prepare_inputs_for_generation = model.base_model.prepare_inputs_for_generation
            model._reorder_cache = model.base_model._reorder_cache
            return model
        else:
            return cls.from_pretrained_old_peft_version(model, model_id, **kwargs)


    @classmethod
    def from_pretrained_old_peft_version(cls, model, model_id, **kwargs):
        # work well in peft@e536616888d51b453ed354a6f1e243fecb02ea08

        # load the config
        config = LoraConfig.from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        # here is the hack
        model = cls(model, config)
        model._reorder_cache = model.base_model._reorder_cache
        # load weights if any
        if os.path.exists(os.path.join(model_id, "adapter_model.bin")):
            filename = os.path.join(model_id, "adapter_model.bin")
        else:
            try:
                filename = hf_hub_download(model_id, "adapter_model.bin")
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {'adapter_model.bin'} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model
