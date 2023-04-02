################################
# one epoch on vi_alpaca_reduced 
################################
python3 finetune.py --data_path 'vi_alpaca_reduced.jsonl' \
    --batch_size=128 --micro_batch_size 4 --num_epochs 3 --output_dir 'chat-gpt-neo-1.3B-3e' \
    # --bf16 False --fp16 True # <= để chạy trên Google Colab (GPUs đời cũ)


#######################
## GPU siêu nhỏ 4G vram
#######################

# python3 finetune.py --data_path 'vi_alpaca_reduced.jsonl' \
# 	--base_model 'truongpdd/vietnews-gpt2' --output_dir 'chat-gpt2-200m-1e' \
# 	--batch_size=128 --micro_batch_size 2 --num_epochs 1 \
# 	--lora_target_modules 'c_proj'

# (transformer): GPT2Model(
#   (wte): Embedding(50257, 768)
#   (wpe): Embedding(1024, 768)
#   (drop): Dropout(p=0.0, inplace=False)
#   (h): ModuleList(
#     (0-11): 12 x GPT2Block(
#       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#       (attn): GPT2Attention(
#         (c_attn): Conv1D()
#         (c_proj): Conv1D()
#         (attn_dropout): Dropout(p=0.0, inplace=False)
#         (resid_dropout): Dropout(p=0.0, inplace=False)
#       )
#       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#       (mlp): GPT2MLP(
#         (c_fc): Conv1D()
#         (c_proj): Conv1D()
#         (act): NewGELUActivation()
#         (dropout): Dropout(p=0.0, inplace=False)
#       )
#     )
#   )
#   (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)