############################
## Cách tạo dữ liệu tổng hợp
############################
[ -f data/vi_merged.jsonl ] || cat data/vi*.jsonl > data/vi_merged.jsonl

##########################################
# Các kịch bản fine-tune với 3060 12G vram
##########################################

## Prefix tuning
#nohup python3 finetune.py --data_path 'data/vi_merged.jsonl' --base_model 'VietAI/gpt-neo-1.3B-vietnamese-news' \
#    --finetune_method 'prefix' --num_virtual_tokens 64  --output_dir 'out/prefix_gpt-neo-1.3B-2e' \
#    --batch_size=128 --micro_batch_size 1 --cutoff_len 512 --num_epochs 2 \
#    --load_in_8bit False --bf16 True &
    # --resume_from_checkpoint 'out/prefix_gpt-neo-1.3B-1e/checkpoint-1600' &
    # Vì model nhỏ nên không cần 8-bit và dùng bf16 để tận dụng tensor cores

## LoRA tuning
python3 finetune.py --data_path 'data/sample.jsonl' --base_model 'VietAI/gpt-neo-1.3B-vietnamese-news' \
    --finetune_method 'lora' --lora_r 16 --lora_alpha 16 --output_dir 'out/lora_gpt-neo-1.3B-1e' \
    --batch_size=128 --micro_batch_size 1 --cutoff_len 256 --num_epochs 1 \
    --load_in_8bit False --bf16 True

#python3 finetune.py --data_path 'data/vi_merged.jsonl' --base_model 'VietAI/gpt-j-6B-vietnamese-news' \
#    --finetune_method 'lora' --lora_r 16 --lora_alpha 16 --output_dir 'out/lora_gpt-j-6B-1e' \
#    --batch_size=128 --micro_batch_size 2 --cutoff_len 512 --num_epochs 1 --load_in_8bit True
