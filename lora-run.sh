################################
# one epoch on vi_alpaca_reduced 
################################
deepspeed lora.py --deepspeed ds_z3_bf16_config.json --data_path 'vi_alpaca_reduced.jsonl' \
	--batch_size=256 --per_device_batch_size 4 --num_epochs 1 --output_dir 'chat-gpt-neo-1.3B-1e' \
	--bf16 False --fp16 True

##############################
## Train on a combined dataset
##############################
# [ -f vi_merged.jsonl ] || cat vi*.jsonl > vi_merged.jsonl

# deepspeed lora.py --deepspeed ds_z3_bf16_config.json --data_path 'vi_merged.jsonl' \
# 	--batch_size=256 --per_device_batch_size 4 --num_epochs 1 --output_dir 'chat-gpt-neo-1.3B-1e'

#CUDA_VISIBLE_DEVICES=2 deepspeed lora.py --deepspeed ds_z3_bf16_config.json --output_dir './chat-gpt-neo-1.3B_merged-1e' --lora_r 16 \
#	--data_path './vi_merged.jsonl' --batch_size=240 --per_device_batch_size 30 --num_epochs 1


#######################
## GPU siêu nhỏ 4G vram
#######################
# deepspeed lora.py --deepspeed ds_z2_fp16_config.json --data_path 'vi_alpaca_reduced.jsonl' \
# 	--base_model 'truongpdd/vietnews-gpt2' --output_dir 'chat-gpt2-200m-1e' \
# 	--batch_size=128 --per_device_batch_size 2 --num_epochs 1 --lora_target_modules 'c_proj' \
# 	--bf16 False --fp16 True


################################
## Kiểm tra trên tập dữ liệu nhỏ
################################
#deepspeed lora.py --deepspeed ds_z3_bf16_config.json --data_path sample.jsonl --val_set_size=20

#deepspeed --num_gpus=4 lora.py --deepspeed ds_z3_bf16_config.json --output_dir './chat-gpt-neo-1.3B-3e' --lora_r 16 \
#   --data_path './sample.jsonl' --batch_size=240 --per_device_batch_size 30 --num_epochs 3 --val_set_size=20
