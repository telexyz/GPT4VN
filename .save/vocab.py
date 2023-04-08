from transformers import AutoTokenizer
BASE_MODEL = "VietAI/gpt-j-6B-vietnamese-news"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
for i in range(60_000):
	x = tokenizer.decode(i)
	print(f"{i}\t\t{x}")