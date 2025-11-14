from transformers import AutoTokenizer
from datasets import load_dataset
import random
import json
import os


#################################################
# Settings 

model_name = "Qwen/Qwen3-30B-A3B"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

dataset_name = "HuggingFaceFW/fineweb-edu"
# "HuggingFaceFW/fineweb-edu", "wikitext", "cais/mmlu"

# Dataset configurations
dataset_configs = {
    "HuggingFaceFW/fineweb": {"config": None, "streaming": True, "split": "train", "has_subjects": False},
    "HuggingFaceFW/fineweb-edu": {"config": None, "streaming": True, "split": "train", "has_subjects": False},
    "wikitext": {"config": "wikitext-103-v1", "streaming": False, "split": "train", "has_subjects": False},
    "cais/mmlu": {"config": "all", "streaming": False, "split": "auxiliary_train", "has_subjects": True}
}

token_lengths = [512, 1024]  # [1, 2, ... , 512, 1024]

num_per_token_length = 1024

#################################################



print("-----------------------------")
print(f"Processing {dataset_name}")
print("-----------------------------")

# Get dataset configuration
config_info = dataset_configs[dataset_name]
config = config_info["config"]
dataset_streaming = config_info["streaming"]
dataset_split = config_info["split"]
has_subjects = config_info["has_subjects"]

if config:
    dataset = load_dataset(dataset_name, config, split=dataset_split, streaming=dataset_streaming)
else:
    dataset = load_dataset(dataset_name, split=dataset_split, streaming=dataset_streaming)

results = {}

for token_length in token_lengths:
    print(f"Collecting samples for token length {token_length}...")
    results[token_length] = []
    
    if has_subjects:  # mmlu
        subjects = list(set(dataset['subject']))
        samples_per_subject = num_per_token_length // len(subjects)
        remainder = num_per_token_length % len(subjects)
        
        for i, subject in enumerate(subjects):
            subject_data = dataset.filter(lambda x: x['subject'] == subject)
            target_samples = samples_per_subject + (1 if i < remainder else 0)
            collected = 0
            
            for item in subject_data.shuffle():
                if collected >= target_samples:
                    break
                    
                # 정답만 추출 (A), B) 등 제거)
                answer_text = item['choices'][item['answer']].split(') ', 1)[-1]
                tokens = tokenizer.encode(answer_text)
                
                if len(tokens) >= token_length:
                    truncated_tokens = tokens[:token_length]
                    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    results[token_length].append(truncated_text)
                    collected += 1
    
    else:  # fineweb, wikitext etc.
        collected = 0
        for item in dataset.shuffle():
            if collected >= num_per_token_length:
                break
                
            text = item['text'] if 'text' in item else str(item)
            tokens = tokenizer.encode(text)
            
            if len(tokens) >= token_length:
                truncated_tokens = tokens[:token_length]
                truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                results[token_length].append(truncated_text)
                collected += 1

print("Collection completed!")

# 결과 출력
for length in token_lengths:
    count = len(results[length])
    print(f"Token length {length}: {count} samples")

# JSON 파일로 저장
output_filename = f"test_dataset_{dataset_name.replace('/', '_')}_{dataset_split}.json"
output_data = {
    "dataset_info": {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "split": dataset_split,
        "config": config,
        "has_subjects": has_subjects
    },
    "token_lengths": token_lengths,
    "num_per_token_length": num_per_token_length,
    "results": {str(length): texts for length, texts in results.items()}
}

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\nDataset saved to: {output_filename}")

print("-----------------------------")
print(f"Completed processing {dataset_name}")
print("-----------------------------")