import json
import os

def load_prompts(batch_size, prompt_len):
    """
    Load prompts from test dataset JSON file.

    Args:
        batch_size (int): Number of prompts to return
        prompt_len (int): Token length of prompts to load

    Returns:
        list[str]: List of prompt strings

    Raises:
        ValueError: If prompt_len not available or batch_size exceeds available data
        FileNotFoundError: If dataset file not found
    """
    json_path = os.path.join("/shared/datasets/test_dataset.json")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if prompt_len is available
    if prompt_len not in data["prompt_length"]:
        prompt_len = 16384
        # raise ValueError(f"Prompt length {prompt_len} not available. Available lengths: {data['prompt_length']}")

    # Get prompts for the specified length
    prompts = data["results"][str(prompt_len)]
    num_available = len(prompts)

    # If batch_size exceeds available data, replicate prompts
    if batch_size > num_available:
        num_repeats = (batch_size + num_available - 1) // num_available  # Ceiling division
        prompts = (prompts * num_repeats)

    return prompts[:batch_size]
