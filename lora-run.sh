[ -f vi_merged.jsonl ] || cat vi*.jsonl > vi_merged.jsonl

# https://huggingface.co/docs/transformers/main_classes/deepspeed
deepspeed --num_gpus=1 lora.py --deepspeed ds_z3_bf16_config.json
