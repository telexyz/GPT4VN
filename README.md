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

> Nếu không tự chạy được thì hãy trải nghiệm với chatbot tại https://discord.gg/fQ9ja2jBR9

## Show me how
```sh
./lora-run.sh
```
Ví dụ trên huấn luyện chỉ dẫn `VietAI/gpt-neo-1.3B-vietnamese-news` với 51 nghìn câu chỉ cần trên GPU 3060 12G vram hoàn tất trong khoảng một giờ cho một epoch. Cùng setting trên huấn luyện trên 4 GPU A100 mất tầm 5 phút. Nên huấn luyện cho tới khi loss ổn định (3-5 epochs).

Tham gia thảo luận tại https://discord.gg/NuYwhH6Kbb để được hướng dẫn chi tiết.
