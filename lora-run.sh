[ -f vi_merged.jsonl ] || cat vi*.jsonl > vi_merged.jsonl

# https://huggingface.co/docs/transformers/main_classes/deepspeed
nohup deepspeed --num_gpus=3 lora.py --deepspeed ds_z3_bf16_config.json &
tail -f nohup.out

