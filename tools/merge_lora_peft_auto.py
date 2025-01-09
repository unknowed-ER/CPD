from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import utils
import torch
import middleware
import prompt
import copy
import re
import tqdm
from torch.nn.parameter import Parameter
try:
    from peft.utils import PromptLearningConfig
except:
    exit('need new peft version: pip install peft==0.3.0')

import warnings
from peft.tuners.lora import Linear8bitLt, Linear, LoraLayer,LoraModel
from peft.utils import transpose
from torch import nn

def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
    if not hasattr(self, 'exist_adapters'):
        self.exist_adapters = []
    self.exist_adapters.append(adapter_name)
    # ----------original logit-----------------------
    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()

    self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
    # Actual trainable parameters
    if r > 0:
        self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
        self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
        self.scaling[adapter_name] = lora_alpha / r
    if init_lora_weights:
        self.reset_lora_parameters(adapter_name)
    self.to(self.weight.device)
    
def set_multi_adapters(self, names, weights, freeze_weights=True):
    self.active_adapters = []
    weights_list = []
    for n,w in zip(names,weights):
        if n not in self.exist_adapters:
            string = f"{n} is not add yet! ignore it. Notice that the weight is still {w}"
            warnings.warn(string)
            continue
        weights_adpater = Parameter(torch.tensor(w))
        weights_adpater.requires_grad = True if not freeze_weights else False
        self.active_adapters.append({'name':n, 'weight':weights_adpater})

def multi_merge(self):

    if self.merged:
        warnings.warn("Already merged. Nothing to do.")
        return
    for item in self.active_adapters:
        adapter_name = item['name']
        weight = item['weight']
        if self.r[adapter_name] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[adapter_name] * weight
            )
    self.merged = True

def multi_unmerge(self):

    if not self.merged:
        warnings.warn("Already unmerged. Nothing to do.")
        return
    for item in self.active_adapters:
        adapter_name = item['name']
        weight = item['weight']
        if self.r[adapter_name] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[adapter_name] * weight
            )
    self.merged = False

def call_module_fn(model, fn_name, **kargs):
    # python call class method by string
    for module in model.modules():
        if isinstance(module, LoraLayer):
            getattr(module, fn_name)(**kargs)

def module_map(self, fn_name, **kargs):
    # python call class method by string
    for module in self.modules():
        if isinstance(module, LoraLayer):
            getattr(module, fn_name)(**kargs)

setattr(Linear, 'update_layer', update_layer) # add_adapter -> _find_and_replace -> update_layer
setattr(Linear, 'set_multi_adapters', set_multi_adapters)
setattr(Linear, 'multi_merge', multi_merge)
setattr(Linear, 'multi_unmerge', multi_unmerge)
setattr(LoraModel, 'module_map', module_map)

# setattr(middleware.InferMiddleWare, 'infer_chat', infer_chat)
logger = utils.set_console_logger('merge')
model_name = "/mnt/wsl/PHYSICALDRIVE2/models/yahma_llama_7b"
key = ['legal','chat']
test_num = 2
lora_path={
    'legal': 'outs/7b-legal-2048-raw',
    'chat': 'outs/7b-sharegpt-4090-2',
    'medical': '',
    'shi': '',
    'insurance': 'test/20230511/insurance_test.json',
    'instruct': '',
}
data_path = {
    'legal': 'test/20230511/legal_test.json',
    'chat': 'test/20230508/test_20230507-format2.json',
    'medical': 'test/20230511/medical_test.json',
    'shi': '',
    'insurance': 'test/20230511/insurance_test.json',
    'instruct': '',
}
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id=0
# TODO: int8  fp16，，fp16 mergeint8 https://github.com/huggingface/peft/issues/444
model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,
    torch_dtype=torch.float16, # default is fp32
    device_map="auto",
    use_auth_token=True
)
# load
model = PeftModel.from_pretrained(model, lora_path[key[0]], adapter_name=key[0])
model.load_adapter(lora_path[key[1]], adapter_name=key[1])
# 
datas = []
for k in key:
    dat = utils.from_json(data_path[k])
    item = [k for k in  dat.keys() if '7b' in k and len(dat[k])!=0][0]
    datas.extend(dat[item][:test_num])
# fp16 only 512
mid = middleware.InferMiddleWare(model, tokenizer)
mid.generation_config['max_new_tokens'] = 512
mid.generation_config['max_memory'] = 512

def weight_module_add():
    # TODO: weight， target_modules
    # weight=0 
    pass

def weight_add():

    # weight
    ans = {
        'model': model_name,
        'lora': {k: lora_path[k] for k in key},
        'data': {k: data_path[k] for k in key},
    }
    for i in [x / 10.0 for x in range(0, 20, 2)]:
        for j in [x / 10.0 for x in range(0, 20, 2)]:
            weight = [i,j]
            names = '-'.join([f'{k}@{w}' for k,w in zip(key,weight)])
            logger.info(f'exist_adapters:{model.base_model.model.model.layers[0].self_attn.q_proj.exist_adapters}')
            model.base_model.module_map('set_multi_adapters',names=key, weights=weight) # call_module_fn(model.base_model, 'set_multi_adapters', names=['Chinese_Alpaca', 'Medical_Alpaca'], weights=[1.6,0.4])
            logger.info(f'active_adapters:{model.base_model.model.model.layers[0].self_attn.q_proj.active_adapters}' )
            logger.info(f'before merge:{model.base_model.model.model.layers[0].self_attn.q_proj.weight}')
            model.base_model.module_map('multi_merge') # call_module_fn(model.base_model, 'multi_merge')
            logger.info(f'after merge:{model.base_model.model.model.layers[0].self_attn.q_proj.weight}')
            result = mid.infer_chat(datas)
            ans[names]=result
            model.base_model.module_map('multi_unmerge') # call_module_fn(model.base_model, 'multi_unmerge')
            logger.info(f'after unmerge:{model.base_model.model.model.layers[0].self_attn.q_proj.weight}')
    utils.to_json( ans, 'weight_test.json')
    #  weights=[0.2,0.6]
    # ，1987，、。</s>
    # 1.。2.、、、，。3.。4.。5.。6.。7.。8.。9.。10.。11.。12.。13.。14.。15.。16.。17.。18.。19.。20.。21.。22.。23.。24.。25.。26.。27.。28.
    #  weights=[0.6,0.4]
    # ，1987。</s>
    # ，，。，。</s>
    # Sefa

def peft_weight_add():
    instruction1 = ""
    instruction2 = ""
    # ：；forwardtarget-module
    # medicalchatv1qv
    model.add_weighted_adapter(adapters=["Medical_Alpaca","Chinese_Alpaca"], weights=[0.2, 0.8], adapter_name='merged')
    model.set_adapter('merged') # model.base_model.model.model.layers[0].self_attn.q_proj.active_adapter
    print(infer(instruction1,None,None)[0])
    print(infer(instruction2,None,None)[0])
    # ("You're welcome.</s>")
    # ('\n\n')

def peft_activate():
    # NOTE peftMOEactivate，medicalqv，chatv1komlp，
    instruction1 = ""
    instruction2 = ""
    print(infer(instruction1,None,None)[0])
    print(infer(instruction2,None,None)[0])
# '1987，。</s>',
# '。，B，。\n\n、。，。</s>', 
    # SINGLE
    model.set_adapter("Medical_Alpaca")
    instruction = "Tell me about alpacas."
    print(infer(instruction1,None,None))
# ('？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？',
# '1。：，，，，，。2。，。\n\n', 
    model.set_adapter("Chinese_Alpaca")
    print(infer(instruction2,None,None))
# '，1987。</s>'
# '！，、、、、。：\n\n1. ：，。，，、。\n2. ：，。，，、。\n3. ：，。，，、。\n\n。，。</s>'
    with model.disable_adapter():
        instruction = ""
        print(infer(instruction1,None,None))
        print(infer(instruction2,None,None))
# 'Your phone number is 1234567890.\n\n'
# '\n\n'

try:
    weight_add()
except:
    import sys,pdb,bdb
    type, value, tb = sys.exc_info()
    if type == bdb.BdbQuit:
        exit()
    print(type,value)
    pdb.post_mortem(tb)
