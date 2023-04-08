## Cách tạo dữ liệu tổng hợp
# [ -f data/vi_merged.jsonl ] || cat data/vi*.jsonl > data/vi_merged.jsonl

## Các kịch bản mẫu để fine-tune
## 1/ LoRA tuning
# python3 finetune.py --data_path 'data/sample.jsonl' --base_model 'VietAI/gpt-j-6B-vietnamese-news' \
#     --finetune_method 'lora' --lora_r 16 --lora_alpha 16 --output_dir 'out/gpt-j-6B-1e' \
#     --batch_size=128 --micro_batch_size 2 --cutoff_len 512 --num_epochs 1

## 2/ Prefix tuning
python3 finetune.py --data_path 'data/sample.jsonl' --base_model 'VietAI/gpt-neo-1.3B-vietnamese-news' \
    --finetune_method 'prefix' --num_virtual_tokens 32  --output_dir 'out/gpt-j-1.3B-1e' \
    --batch_size=128 --micro_batch_size 2 --cutoff_len 512 --num_epochs 1
