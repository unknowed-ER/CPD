import os
#os.environ['XDG_CACHE_HOME'] = '/data/.cache'
# ，
import utils
import torch
import huggingface_hub


# The XDG_CACHE_HOME is not used if HF_HOME is set. If it wasn't set as above then HF_HOME defaults to $XDG_CACHE_HOME/huggingface
# os.environ['HF_HOME'] = '/data/.cache/huggingface'
# os.environ['HF_DATASETS_CACHE'] = '/data/.cache/huggingface/dataset'
# os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/huggingface/models'

print(huggingface_hub.constants.HUGGINGFACE_HUB_CACHE)
utils.MODEL_DIR='outs/models'
# utils.download_hf('bert-base-uncased',f'{utils.MODEL_DIR}/bert-base-uncased',type='raw',torch_dtype=torch.float16)
utils.download_hf('gpt2-xl',f'{utils.MODEL_DIR}/gpt2-xl', type='causal', torch_dtype=torch.float16)
# utils.download_hf('meta-llama/Llama-2-7b-hf', f'{utils.MODEL_DIR}/Llama-2-7b-hf', type='causal', torch_dtype=torch.float16)
# utils.MODEL_DIR='/data/models'
# utils.download_hf('bert-base-uncased')
# utils.download_hf('microsoft/DialoGPT-medium',f'{utils.MODEL_DIR}/microsoft/DialoGPT-medium', type='causal', torch_dtype=torch.float16)
# utils.download_hf('imone/LLaMA2_13B_with_EOT_token', f'{utils.MODEL_DIR}/imone_LLaMA2_13B_EOT', type='causal', torch_dtype=torch.float16)
# utils.download_hf('meta-llama/Llama-2-70b-chat-hf', f'{utils.MODEL_DIR}/Llama-2-70b-chat-hf', type='causal', torch_dtype=torch.float16)
# utils.download_hf('meta-llama/Llama-2-7b-chat-hf', f'{utils.MODEL_DIR}/Llama-2-7b-chat-hf', type='causal', torch_dtype=torch.float16)
# utils.download_hf('yahma/llama-7b-hf', f'{utils.MODEL_DIR}/yahma_llama_7b/',type='causal', torch_dtype=torch.float16)
# utils.download_hf('yahma/llama-13b-hf', f'{utils.MODEL_DIR}/yahma_llama_13b/',type='causal', torch_dtype=torch.float16)
# utils.download_hf('meta-llama/Llama-2-13b-hf', f'{utils.MODEL_DIR}/Llama-2-13b-hf', type='causal', torch_dtype=torch.float16)
# utils.download_hf('huggyllama/llama-30b')
