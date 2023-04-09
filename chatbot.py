#!/usr/bin/python3
import sys
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import make_prompt

BASE_MODEL = "VietAI/gpt-j-6B-vietnamese-news"
PEFT_WEIGHTS = "tiendung/chat-gpt-j-6B-t"
load_in_8bit = True

BASE_MODEL = "VietAI/gpt-neo-1.3B-vietnamese-news"
PEFT_WEIGHTS = "tiendung/prefix_gpt-neo-1.3B-1e"
load_in_8bit = False

if torch.cuda.is_available():
    device = "cuda"
    device_map = {'': 0}
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
        model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, torch_dtype=torch.float16, device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map=device_map)
        model = PeftModel.from_pretrained(model, PEFT_WEIGHTS, device_map=device_map)
else:
    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(model, PEFT_WEIGHTS)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.eval()
if torch.__version__ >= "2": # tăng tốc
    model = torch.compile(model)

def get_answer(q, max_new_tokens=196, skip_tl=False):
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
    output = origin_output.split("###")[2]
    try:
        k = output.index(":")
        if k < 10: output = output[k+1:]
    except:
        output = output
    print(f"\n- - -{origin_output}- - -\n")
    return output.strip()

def main():
    print("\n")
    while True:
        query = input("\nBạn：")
        print(f"Bot: {get_answer(query)}")

if __name__ == "__main__": main()
