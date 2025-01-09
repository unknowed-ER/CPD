source scripts/platform.sh
source activate lla

export CUDA_VISIBLE_DEVICES=0

# jurigged -v chat.py \
# python chat.py \
# --model-path $MODEL_DIR/Llama-2-7b-hf/ \
# --dtype int8 \
# --gradio \
# --share-link \

# lora_model_list=( 'outs/instuct_chat_50k_lmhead' )
# for LORA_PATH in "${lora_model_list[@]}"; do
#     python chat_cmd.py \
#     --model_path $MODEL_DIR/Llama-2-7b-hf/ \
#     --lora-path $LORA_PATH \
#     --dtype int8 
# done

DEBUG=1 
python chat.py \
--model-path $MODEL_DIR/Llama-2-7b-hf/ \
--checkpoint-dir /data/outs/collie/tp8-dp1-pp1-zero3/epoch_1 \
--dtype fp16 \
--prompt-type chat \
--data-path outs/chat_test.jsonl \
--out-path outs/instruct_tuning/tp8-dp1-pp1-zero3-fp16.jsonl