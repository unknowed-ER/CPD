
CHECKPOINTS=('754' '1508' '2262' '3017' '3771' '4525' '5279' '6034' '6788' '7542' '8297' '9048')
#CHECKPOINTS=('35' '71' '107' '143' '178' '214' '250' '286' '321' '357' '393' '420')
#CHECKPOINTS=('107' '214' '321' '429' '536' '643' '750' '858' '965' '1072' '1180' '1284')
#LORA_PATH="outs/ESConv_llama2_7b_causal_kl/checkpoint-"
#OUTPUT_PATH="output/ESConv_llama2_7b_causal_kl_checkpoint_"
#OUTPUT_PATH="output/ESConv_qwen_14b_wo_awq_checkpoint_"
#OUTPUT_PATH="output/ESConv_llama2_7b_random_checkpoint_"
OUTPUT_PATH="output/msc_llama2_7b_checkpoint_"
str1=".jsonl"

for i in "${!CHECKPOINTS[@]}"; do
#  echo CUDA_VISIBLE_DEVICES=${TOT_CUDA} python \
#  get_causal_data.py --model_path $BASE_MODEL --lora_path ${LORA_PATH}${CHECKPOINTS[i]} --data_path datasets/CGDIALOG_new/ESConv.jsonl --output_path ${OUTPUT_PATH}${CHECKPOINTS[i]}${str1} --test_batch_size 4 --eval 1
#  CUDA_VISIBLE_DEVICES=${TOT_CUDA} python \
#  get_causal_data.py --model_path $BASE_MODEL --lora_path ${LORA_PATH}${CHECKPOINTS[i]} --data_path datasets/CGDIALOG_new/ESConv.jsonl --output_path ${OUTPUT_PATH}${CHECKPOINTS[i]}${str1} --test_batch_size 4 --eval 1

  echo checkpoint ${CHECKPOINTS[i]}
  python evaluate.py --data_path ${OUTPUT_PATH}${CHECKPOINTS[i]}${str1}
done