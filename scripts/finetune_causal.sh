#source scripts/platform.sh
#DATA_PATH="./datasets/msc/msc_train.jsonl"
#OUTPUT_PATH="./outs/msc_llama2_7b_causal"
DATA_PATH="./datasets/ESConv/ESConv_train.jsonl"
OUTPUT_PATH="./outs/ESConv_llama2_7b_causal_kl"
MODEL_PATH="/model/llama-2-7b-chat-hf/"
#DEBUG="True"
# OUTPUT_PATH="./outs/msc_longchat_7b_no_pos"
# MODEL_PATH="/model/longchat-7b-v1.5-32k/"
# DATA_PATH="instruct_chat_50k.jsonl"
# OUTPUT_PATH="outs/instuct_chat_50k"
# MODEL_PATH="yahma/llama-7b-hf"

TOT_CUDA="2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT=$(shuf -i25000-30000 -n1)

#accelerate launch --config_file config/simple.yaml \
echo CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT \
finetune_causal.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--output_path $OUTPUT_PATH \
--num_epoch 12 \
--micro_batch 2 \
--total_batch 128 \
--dtype 'bf16' \
--log_steps 1 \
--eval_steps 0 \
--warmup_ratio 0.05 \
--save_strategy 'epoch' \
--save_steps 100 \
--test_size 0 \
--cutoff_len 4096 \
--prompt_type "persona_chat" \
--use_data_instruction \
--wandb \
--resume