#!/usr/bin/python3
import sys
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import make_prompt

#BASE_MODEL = "VietAI/gpt-j-6B-vietnamese-news"
#LORA_WEIGHTS = "tiendung/chat-gpt-j-6B-2e"
BASE_MODEL = "VietAI/gpt-neo-1.3B-vietnamese-news"
LORA_WEIGHTS = "tiendung/chat-gpt-neo-1.3B-alpaca-1e"

if torch.cuda.is_available():
    device = "cuda"
    device_map = {'': 0}
else:
    device = "cpu"
    device_map = "auto"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, \
    torch_dtype=torch.float16, device_map=device_map)

model = PeftModel.from_pretrained(model, LORA_WEIGHTS, \
    torch_dtype=torch.float16, device_map=device_map)

model.eval()
if torch.__version__ >= "2": model = torch.compile(model) # tăng tốc

def get_answer(q, max_new_tokens=320, skip_tl=False):
    input_ids = tokenizer(make_prompt(q), return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids=input_ids,
            max_length=len(input_ids) + max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_k=20,
            repetition_penalty=1.2,
            eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.eos_token_id,
        )
    origin_output = tokenizer.batch_decode(gen_tokens)[0]
    output = origin_output.split("### Response")[1]
    if output[0] == ":": output = output[1:]
    return output.strip()

def main():
    print("\n")
    while True:
        query = input("\nBạn：")
        print(f"Bot: {get_answer(query)}")

if __name__ == "__main__": main()
