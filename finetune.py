# Chỉnh sửa từ https://github.com/tloen/alpaca-lora/blob/main/finetune.py; Apache License 2.0
import os
try: os.environ["CUDA_VISIBLE_DEVICES"]
except: os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from peft import (  # noqa: E402
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM # noqa: F402

def train(
    data_path: str = "./data/vi_merged.jsonl",
    base_model: str = "VietAI/gpt-neo-1.3B-vietnamese-news",
    output_dir: str = "./chat-gpt-neo-1.3B",
    # base_model: str = "VietAI/gpt-j-6B-vietnamese-news",
    # output_dir: str = "./chat-gpt-j-6B-1e",

    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 200,

    ## Select finetune method
    finetune_method: str = "", # lora prefix

    # prefix tuning hyperparams
    # Tham khảo https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prefix_tuning_clm.ipynb
    num_virtual_tokens: int = 32,

    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj k_proj v_proj", # gpt-3

    # llm hyperparams
    bf16: bool = False, # whether to use bf16 (preferred on A100's).
    load_in_8bit: bool = True, # 8 bit sẽ giảm vram nhưng chậm tốc độ huấn luyện đi nhiều lần
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    # In ra các tham số chung
    print("\nFINE-TUNE METHOD:", finetune_method)
    print(
        f"Mô hình được finetune và các tham số chung:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )

    if finetune_method == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules.split(), # phân tách str thành list
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(
            f"Training LoRA model with params:\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
        )
    elif finetune_method == "prefix":
        config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens
        )
        print(
            f"Training Prefix-tuning model with params:\n"
            f"num_virtual_tokens: {num_virtual_tokens}\n"
        )
    else:
        assert False, "Hiện tại chỉ hỗ trợ lora và prefix tuning"

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='VietAI/gpt-j-6B-vietnamese-news'"

    gradient_accumulation_steps = batch_size // micro_batch_size
    if load_in_8bit: bf16 = False # nếu load 8 bit thì buộc phải dùng bf16
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp: # huấn luyện đa GPUs
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    if finetune_method == "lora":
        print(model.state_dict) # in ra model state để lựa chọn cho lora

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0 # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
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
        return tokenize(full_prompt)

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)

    if data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None


    training_args = transformers.TrainingArguments(
            fp16=(not bf16), # tốt cho GPUs đời cũ và training 8-bit
            bf16=bf16, # tốt cho GPUs đời mới và không dùng 8-bit
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="none", # không log vào wandb (default option)
            run_name=None,
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

if __name__ == "__main__":
    fire.Fire(train)
