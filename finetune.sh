# [ -f data/vi_merged.jsonl ] || cat data/vi*.jsonl > data/vi_merged.jsonl

## Mặc định là lora finetune
python3 finetune.py --data_path 'data/sample.jsonl' --base_model 'VietAI/gpt-j-6B-vietnamese-news' \
    --batch_size=128 --micro_batch_size 2 --cutoff_len 512 --num_epochs 1 --output_dir 'out/gpt-j-6B-1e'

## Prefix finetune
python3 finetune.py --data_path 'data/sample.jsonl' --base_model 'VietAI/gpt-neo-1.3B-vietnamese-news' \
    --finetune 'prefix' --num_virtual_tokens 32 \
    --batch_size=128 --micro_batch_size 2 --cutoff_len 512 --num_epochs 1 --output_dir 'out/gpt-j-1.3B-1e'

## Thử gpt2 nhưng bị lỗi với aten cuda
# python3 finetune.py --data_path 'data/sample.jsonl' --base_model 'NlpHUST/gpt2-vietnamese' \
#     --batch_size=128 --micro_batch_size 2 --cutoff_len 256 --num_epochs 1 \
#     --lora_target_modules 'c_proj' --output_dir 'out/gpt2-1e'
