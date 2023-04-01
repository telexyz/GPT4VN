[ -f vi_merged.jsonl ] || cat vi*.jsonl > vi_merged.jsonl

#deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json --data_path vi_alpaca_reduced.jsonl

CUDA_VISIBLE_DEVICES=2 deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json --output_dir './chat-gpt-neo-1.3B_merged-1e' --lora_r 16 \
	--data_path './vi_merged.jsonl' --batch_size=256 --per_device_batch_size 32 --num_epochs 1

## Kiểm tra trên tập dữ liệu nhỏ
#deepspeed --num_gpus=4 lora.py --deepspeed ds_z3_bf16_config.json --output_dir './chat-gpt-neo-1.3B-3e' --lora_r 16 \
#        --data_path './sample.jsonl' --batch_size=240 --per_device_batch_size 30 --num_epochs 3 --val_set_size=20
#
#deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json --data_path sample.jsonl --val_set_size=20
