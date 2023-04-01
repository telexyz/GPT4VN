[ -f vi_merged.jsonl ] || cat vi*.jsonl > vi_merged.jsonl

deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json --data_path vi_alpaca_reduced.jsonl

#deepspeed --num_gpus=3 lora.py --deepspeed ds_z3_bf16_config.json --output_dir './chat-gpt-neo-1.3B-5e' --lora_r 16 \
#	--data_path './vi_alpaca_reduced.jsonl' --batch_size 180 --per_device_batch_size 20 --num_epochs 2

#deepspeed --num_gpus=3 lora.py --deepspeed ds_z3_bf16_config.json --output_dir './chat-gpt-neo-1.3B-3e' --lora_r 16 \
#        --data_path './sample.jsonl' --batch_size 180 --per_device_batch_size 20 --num_epochs 3 --val_set_size=20

# deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json --data_path sample.jsonl --val_set_size=20 --resume_from_checkpoint checkpoint-200
