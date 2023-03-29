Trải nghiệm với chatbot tại https://discord.gg/fQ9ja2jBR9

https://user-images.githubusercontent.com/8133/228418280-ba026ee4-11ef-4c8e-9edf-cd90ba2dfd1c.mp4

# Dữ liệu chỉ dẫn để biến mô hình ngôn ngữ Việt thành chatbot

- `alpaca_vi.txt`: dịch từ [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) bởi https://www.kaggle.com/datasets/iambestfeeder

- `daily_dialog_vi.txt`: dịch từ [daily_dialog](https://huggingface.co/datasets/daily_dialog) bởi https://www.kaggle.com/datasets/iambestfeeder

- `vi_gpt4all_reduced_*.jsonl`: ~173k dịch từ [gpt4all](https://github.com/nomic-ai/gpt4all), loại bớt và dịch bởi Tuộc và https://github.com/binhvq

- `vi_alpaca_reduced.jsonl`: ~51k chỉ dẫn dịch từ [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned), lược bớt và dịch bởi Tuộc và https://github.com/binhvq

Để tạo một file huấn luyện chung dùng lệnh (__tổng 224k chỉ dẫn__):
```sh
cat *vi*.jsonl > vi_merged.jsonl
```

## Show me results
Các bạn trong nhóm mình đã fine-tune thành công vietai/gpt-j-vietnews tại đây https://huggingface.co/hoaiht/vietnamese-alpaca-lora-gpt-j/tree/main

## Show me how
Tham gia thảo luận tại https://discord.gg/NuYwhH6Kbb để được hướng dẫn nhé. Nhóm sẽ release code sau.
