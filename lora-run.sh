[ -f vi_merged.jsonl ] || cat vi*.jsonl > vi_merged.jsonl

deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json --data_path vi_alpaca_reduced.jsonl

# deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json --data_path sample.jsonl --val_set_size=20 --resume_from_checkpoint checkpoint-200
