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
    # Get the JSON file path (assuming it's in the same directory)
    json_path = os.path.join(os.path.dirname(__file__), "test_dataset_HuggingFaceFW_fineweb_train.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if prompt_len is available
    if prompt_len not in data["token_lengths"]:
        raise ValueError(f"Prompt length {prompt_len} not available. Available lengths: {data['token_lengths']}")
    
    # Get prompts for the specified length
    prompts = data["results"][str(prompt_len)]
    num_available = len(prompts)
    
    # Check if batch_size exceeds available data
    if batch_size > num_available:
        raise ValueError(f"Batch size {batch_size} exceeds available data ({num_available}) for length {prompt_len}")
    
    return prompts[:batch_size]
