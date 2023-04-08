[ -f data/vi_merged.jsonl ] || cat data/vi*.jsonl > data/vi_merged.jsonl

python3 finetune.py --data_path 'data/vi_merged.jsonl' --base_model 'VietAI/gpt-j-6B-vietnamese-news' \
    --batch_size=128 --micro_batch_size 2 --cutoff_len 512 --num_epochs 1 --output_dir 'chat-gpt-j-6B-merged-1e'
