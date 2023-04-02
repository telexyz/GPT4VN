# GPT4VN

Hãy biến mô hình ngôn ngữ thành chatbot

https://user-images.githubusercontent.com/8133/228418280-ba026ee4-11ef-4c8e-9edf-cd90ba2dfd1c.mp4

## Dữ liệu chỉ dẫn và hội thoại

- `alpaca_vi.txt`: dịch từ [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) bởi [Iambestfeed](https://github.com/Iambestfeed)

- `daily_dialog_vi.txt`: dịch từ [daily_dialog](https://huggingface.co/datasets/daily_dialog) bởi [Iambestfeed](https://www.kaggle.com/datasets/iambestfeeder)

- `vi_gpt4all_reduced_*.jsonl`: ~173k lược bớt và dịch từ [gpt4all](https://github.com/nomic-ai/gpt4all) và dịch bởi Tuộc và [binhvq](https://github.com/binhvq)

- `vi_alpaca_reduced.jsonl`: ~51k chỉ dẫn lược bớt và dịch từ [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) bởi Tuộc và [binhvq](https://github.com/binhvq)

Để tạo một file huấn luyện chung dùng lệnh:
```sh
cat vi*.jsonl > vi_merged.jsonl
```

## Show me the results

```sh
./chatbot.sh
```

![vietnam-chatbot](https://user-images.githubusercontent.com/8133/229118963-e34d4dd6-b1ba-4307-9453-043c5afdb979.png)

> TRẢI NGHIỆM VỚI CHATBOT TẠI https://discord.gg/fQ9ja2jBR9

## Show me how
```sh
python3 finetune.py --data_path 'vi_alpaca_reduced.jsonl' \
    --batch_size=128 --micro_batch_size 2 --num_epochs 1 --output_dir 'chat-gpt-neo-1.3B-1e'
```

Ví dụ trên huấn luyện chỉ dẫn `VietAI/gpt-neo-1.3B-vietnamese-news` với 51 nghìn câu trên GPU 3060 12G vram hoàn tất trong khoảng một giờ cho một epoch.

Có thể chạy thử với google colab tại https://colab.research.google.com/drive/11XSZkOfoPbFIIGAs9gRgMuLVQ9mJBPIi nhưng tốc độ huấn luyện chậm đi 4 lần so với 3060 :(

> THAM GIA THẢO LUẬN TẠI https://discord.gg/NuYwhH6Kbb
