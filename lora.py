# - Rút gọn từ https://github.com/tloen/alpaca-lora/blob/main/finetune.py
# - Hỗ trợ deepspeed:
#   - giảm vram cho phép 1.3b model có thể LoRA trên 12G vram
#   - huấn luyện nhanh hơn trên nhiều GPUs => !!! Đang bị lỗi save model !!!
#   - ds config https://github.com/databrickslabs/dolly/blob/master/config/ds_z3_bf16_config.json

import os, sys, fire, torch, transformers
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, \
    prepare_model_for_int8_training, set_peft_model_state_dict

os.environ["TOKENIZERS_PARALLELISM"] = "false" # turn off warning

def train(
    ## Use by deepspeed
    deepspeed,
    batch_size=256,
    per_device_batch_size=4,
    local_rank=True,
    fp16=False,
    bf16=True, # Whether to use bf16 (preferred on A100's).
    gradient_checkpointing=False,
    data_path: str = "./vi_alpaca_reduced.jsonl",
    base_model: str = "VietAI/gpt-neo-1.3B-vietnamese-news",
    output_dir: str = "./chat-gpt-neo-1.3B",
    # training hyperparams
    num_epochs: int = 3,
    lr: float = 5e-5,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    # llm hyperparams
    resume_from_checkpoint: str = None,  # None or string: either training checkpoint or final adapter
):
    if isinstance(lora_target_modules, str):
        lora_target_modules = lora_target_modules.split()
    if fp16 == "True": fp16 = True
    if bf16 == "True": bf16 = True
    if fp16 == "False": fp16 = False
    if bf16 == "False": bf16 = False

    print(
        f"Training LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"per_device_batch_size: {per_device_batch_size}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert base_model
    gradient_accumulation_steps = batch_size // per_device_batch_size

    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained( base_model,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = ( 0 ) # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, \
            max_length=cutoff_len, padding=False, return_tensors=None)
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    data = load_dataset("json", data_files=data_path)
    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin") # only LoRA model
            resume_from_checkpoint = ( False ) # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()
    # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    training_args = transformers.TrainingArguments(
            deepspeed=deepspeed,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            bf16=bf16,
            learning_rate=lr,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            report_to="none",
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)

from prompt import make_prompt
def generate_prompt(data_point):
    question = data_point["prompt"].strip()
    answer = data_point["response"].strip()
    return f"{make_prompt(question)}\n{answer}"

if __name__ == "__main__": fire.Fire(train)
