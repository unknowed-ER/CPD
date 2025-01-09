TOT_CUDA="3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT=$(shuf -i25000-30000 -n1)


# ——————————————————Qwen-14B————————————————————————
BASE_MODEL="/model/Qwen-14B-Chat"
DATA_PATH="datasets/msc/msc_test.jsonl"


OUTPUT_PATH="output/msc_qwen_14b_raw.jsonl"
BASE_MODEL="/model/Qwen-14B-Chat-Int4-AWQ"
echo CUDA_VISIBLE_DEVICES=${TOT_CUDA} python \
generate.py --model_path $BASE_MODEL --use_local 0 --data_path $DATA_PATH --output_path $OUTPUT_PATH --prompt_type persona_chat --cutoff_len 4096 --test_batch_size 1 --kind raw --vllm
CUDA_VISIBLE_DEVICES=${TOT_CUDA} python \
generate.py --model_path $BASE_MODEL --use_local 0 --data_path $DATA_PATH --output_path $OUTPUT_PATH --prompt_type persona_chat --cutoff_len 4096 --test_batch_size 1 --kind raw --vllm

#LORA_PATH="outs/msc_llama2_7b/checkpoint-"
#OUTPUT_PATH="output/msc_llama2_7b_checkpoint_"

# ——————————————————LLama2-7b————————————————————————
#BASE_MODEL="/model/llama-2-7b-chat-hf"
#DATA_PATH="datasets/msc/msc_test.jsonl"

#LORA_PATH="\"\""
#OUTPUT_PATH="output/msc_llama2_7b_raw.jsonl"

#LORA_PATH="outs/msc_llama2_7b/checkpoint-"
#OUTPUT_PATH="output/msc_llama2_7b_checkpoint_"

#KIND="raw"

#
#echo CUDA_VISIBLE_DEVICES=${TOT_CUDA} python \
#generate.py --model_path $BASE_MODEL --lora_path "" --use_local 0 --data_path $DATA_PATH --output_path $OUTPUT_PATH --prompt_type persona_chat --cutoff_len 4096 --test_batch_size 1 --kind raw
#CUDA_VISIBLE_DEVICES=${TOT_CUDA} python \
#generate.py --model_path $BASE_MODEL --lora_path "" --use_local 0 --data_path $DATA_PATH --output_path $OUTPUT_PATH --prompt_type persona_chat --cutoff_len 4096 --test_batch_size 1 --kind raw

# finetune generation
#CHECKPOINTS=('754' '1508' '2262' '3017' '3771' '4525' '5279' '6034' '6788' '7542' '8297' '9048')
#str1=".jsonl"
#
#for i in "${!CHECKPOINTS[@]}"; do
#  echo checkpoint ${CHECKPOINTS[i]}
#  echo CUDA_VISIBLE_DEVICES=${TOT_CUDA} python generate.py \
# --model_path $BASE_MODEL --lora_path ${LORA_PATH}${CHECKPOINTS[i]} --use_local 0 --data_path $DATA_PATH --output_path ${OUTPUT_PATH}${CHECKPOINTS[i]}${str1} --prompt_type persona_chat --cutoff_len 4096 --test_batch_size 1 --kind $KIND
#  CUDA_VISIBLE_DEVICES=${TOT_CUDA} python generate.py \
# --model_path $BASE_MODEL --lora_path ${LORA_PATH}${CHECKPOINTS[i]} --use_local 0 --data_path $DATA_PATH --output_path ${OUTPUT_PATH}${CHECKPOINTS[i]}${str1} --prompt_type persona_chat --cutoff_len 4096 --test_batch_size 1 --kind $KIND
#done
