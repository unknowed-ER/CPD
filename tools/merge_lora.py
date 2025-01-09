from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import utils
import torch
import middleware
import prompt
import copy
import json
import re
import os
import tqdm
try:
    from peft.utils import PromptLearningConfig
except:
    exit('need new peft version: pip install peft==0.3.0')
import argparse
import warnings
from peft.tuners.lora import Linear8bitLt, Linear, LoraLayer,LoraModel
from peft.utils import transpose
from peft import set_peft_model_state_dict
from torch import nn
from research.hackpeft import _prepare_for_multi_lora_infer
from research.config import ATTR_DICT as attr_dict



_prepare_for_multi_lora_infer()
logger = utils.set_console_logger('merge')
model_name = f"{utils.MODEL_DIR}/yahma_llama_7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id=0
# TODO: int8  fp16，，fp16 mergeint8 https://github.com/huggingface/peft/issues/444
INT8 =False
model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=INT8,
    torch_dtype=torch.float16, # default is fp32
    device_map="auto",
    use_auth_token=True
)
#  load all
for i,key in enumerate(attr_dict.keys()):
    if i == 0:
        model = utils.StreamPeftGenerationMixin.from_pretrained(model, attr_dict[key]['lora_path'], adapter_name=key)
    else:
        model.load_adapter(attr_dict[key]['lora_path'], adapter_name=key)
# NOTE that default is attr_dict key 0 !
logger.info(f'exist_adapters:{model.base_model.model.model.layers[0].self_attn.q_proj.exist_adapters}')

# fp16 only 512
mid = middleware.InferMiddleWare(model, tokenizer)
mid.generation_config['max_new_tokens'] = 512
mid.generation_config['max_memory'] = 512

def multi_attr(keylist, weightlist,file_name, repeat_num=1):
    mid.generation_config['max_new_tokens'] = 128
    ans = {}
    if repeat_num > 1:
        mid.generation_config['do_sample'] = True
    for keys in keylist:
        for weights in weightlist:
            attrs = [attr_dict[key]['attribute'] for key in keys]
            model.base_model.module_map('set_multi_adapters',names=keys, weights=weights)
            logger.info(f'active_adapters:{model.base_model.model.model.layers[0].self_attn.q_proj.active_adapters}' )

            logger.debug(f'before merge:{model.base_model.model.model.layers[0].self_attn.q_proj.weight}')
            model.base_model.module_map('multi_merge')            
            logger.debug(f'after merge:{model.base_model.model.model.layers[0].self_attn.q_proj.weight}')

            name = '-'.join([f'{k}@{w}' for k,w in zip(attrs,weights)])
            ans[name] = []
            for prefix in ['In summary','This essay discusses','Views on','The connection','Foundational to this is', 'To review,','In brief,','An illustration of','Furthermore,','The central theme',
            'To conclude,','The key aspect','Prior to this','Emphasised are','To summarise',
            'The relationship','More importantly,','It has been shown','The issue focused on','In this essay',
            'Once upon a time','The book','The chicken','The city','The country',
            'The horse','The lake','The last time','The movie','The painting',
            'The pizza','The potato','The president of the country','The road','The year is 1910']: 
                # 35 generate 5 different sentence??   
                for _ in range(repeat_num):
                    output, history = mid.infer_turn(prefix, prompt_type='attribute_autoencoder',extra=attrs,use_typewriter=True)
                    ans[name].append(output)
                
            model.base_model.module_map('multi_unmerge')
            logger.debug(f'after unmerge:{model.base_model.model.model.layers[0].self_attn.q_proj.weight}')
        utils.to_json( ans, file_name)
    utils.to_json( ans, file_name)

def single_control_test():
    mid.generation_config['max_new_tokens'] = 128
    mid.generation_config['do_sample'] = True
    mid.generation_config['num_beam'] = 20
    keys = []
    # for aa in ['IMDb_sentiment_0','IMDb_sentiment_1']:
    for bb in ['MultiControl_toxic_1']:
        keys.append([bb])
    # for cc in ['AGNews_topic_0','AGNews_topic_1','AGNews_topic_2','AGNews_topic_3']:
    weights=[[1]]
    multi_attr(keys, weights, file_name=f'/home/lzy/Chinese-Vicuna/test/20230518/{bb}.json',repeat_num=5,)

def multi_control_test():
    mid.generation_config['max_new_tokens'] = 128
    mid.generation_config['do_sample'] = True
    mid.generation_config['num_beam'] = 20
    keys = []
    for aa in ['IMDb_sentiment_0','IMDb_sentiment_1']:
        for bb in ['MultiControl_toxic_1']:
            for cc in ['AGNews_topic_0','AGNews_topic_1','AGNews_topic_2','AGNews_topic_3']:
                keys.append([aa,bb,cc])
    weights=[[0.7,0.7,0.7]]
    multi_attr(keys, weights, file_name=f'/home/lzy/Chinese-Vicuna/test/20230518/multi_attr.json',repeat_num=5)

def weight_control_search():
    mid.generation_config['max_new_tokens'] = 128
    mid.generation_config['do_sample'] = True
    mid.generation_config['num_beam'] = 10
    keys = []
    for aa in ['IMDb_sentiment_0','IMDb_sentiment_1']:
        for bb in ['MultiControl_toxic_0','MultiControl_toxic_1']:
            keys.append([aa,bb])
    weights = []
    for i in [x / 10.0 for x in range(0, 20, 2)]:
        for j in [x / 10.0 for x in range(0, 20, 2)]:
            weights.append([i,j])
    multi_attr(keys, weights, repeat_num=5)

def imdb_epoch_search():
    for file in os.listdir('IMDb.sentiment.1'):
        # filename = os.path.join(f'IMDb.sentiment.1/{file}','adapter_model.bin')
        # NOTE change weight
        model.load_adapter(f'IMDb.sentiment.1/{file}', adapter_name='IMDb_sentiment_1')
        # adapters_weights = torch.load(
        #     filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # )
        # set_peft_model_state_dict(model, adapters_weights, adapter_name='IMDb_sentiment_1')
        mid.generation_config['max_new_tokens'] = 128
        mid.generation_config['do_sample'] = True
        mid.generation_config['num_beam'] = 20
        keys = []
        for aa in ['IMDb_sentiment_0','IMDb_sentiment_1']:
            for bb in ['MultiControl_toxic_1']:
                for cc in ['AGNews_topic_0','AGNews_topic_1','AGNews_topic_2','AGNews_topic_3']:
                    keys.append([aa,bb,cc])
        weights=[[0.7,0.7,0.7]]
        if os.path.exists(f'/home/lzy/Chinese-Vicuna/test/20230518/multi_attr_{file}.json'):
            continue
        multi_attr(keys, weights, file_name=f'/home/lzy/Chinese-Vicuna/test/20230518/multi_attr_{file}.json',repeat_num=5)

def yelp_epoch_search():
    mid.generation_config['max_new_tokens'] = 128
    mid.generation_config['do_sample'] = True
    mid.generation_config['num_beam'] = 2
    keys = []
    for aa in ['yelp_sentiment_0','yelp_sentiment_1']:
        for bb in ['yelp_tense_0','yelp_tense_1','yelp_tense_2']:
            keys.append([aa,bb])
    weights = [[0.7,0.7]]
    import eval
    import argparse
    for dir in os.listdir('/home/lzy/Chinese-Vicuna/outs/attribute/yelp_sentiment.0'):
        if 'checkpoint' not in dir:
            continue
        epoch=dir.split('-')[-1]
        if os.path.exists(f'/home/lzy/Chinese-Vicuna/test/20230518/multi_attr_yelp_{epoch}.json'):
            continue
        for key in ['yelp_sentiment_0','yelp_sentiment_1','yelp_tense_0','yelp_tense_1','yelp_tense_2']:
            print(f"{attr_dict[key]['lora_path']}/{dir}")
            model.load_adapter(f"{attr_dict[key]['lora_path']}/{dir}", adapter_name=key)
        multi_attr(keys, weights, file_name=f'/home/lzy/Chinese-Vicuna/test/20230518/multi_attr_yelp_{epoch}.json',repeat_num=1)
        eval.LatentOpsTest(argparse.Namespace(data_path=f'/home/lzy/Chinese-Vicuna/test/20230518/multi_attr_yelp_{epoch}.json', format='Ours', save_path='test/20230518', task_name='LatentOps', ids='0,0;0,1;0,2;1,0;1,1;1,2;-1,-1'))



