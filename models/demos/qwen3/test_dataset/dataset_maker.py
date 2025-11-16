from transformers import AutoTokenizer
from datasets import load_dataset
import json

tokenizer = AutoTokenizer.from_pretrained("/shared/models/Qwen3-30B-A3B")

prompt_lengths = [128, 16384]
max_prompt_length = max(prompt_lengths)
batch_size = 8192

dataset = load_dataset("/shared/datasets/fineweb/sample/10BT", split="train", streaming=False)

results = {prompt_length: [] for prompt_length in prompt_lengths}

text_buffer = ""
collected = 0

for item in dataset:
    if collected >= batch_size:
        break
        
    text = item['text'] if 'text' in item else str(item)
    text_buffer += " " + text
    tokens = tokenizer.encode(text_buffer)
    
    if len(tokens) >= max_prompt_length:
        if collected >= batch_size:
            break

        for prompt_length in prompt_lengths:
            truncated_tokens = tokens[:prompt_length]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            results[prompt_length].append(truncated_text)
        
        text_buffer = ""
        collected += 1

    if collected % 100 == 0:
        print(f"Collected {collected} samples")

output_filename = "test_dataset.json"
output_data = {
    "dataset_info": {
        "dataset_name": "fineweb",
        "model_name": "Qwen3-30B-A3B",
        "split": "train",
    },
    "prompt_length": prompt_lengths,
    "batch_size": batch_size,
    "results": {str(length): texts for length, texts in results.items()}
}

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)