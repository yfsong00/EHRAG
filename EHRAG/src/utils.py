from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
import httpx
from openai import OpenAI
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

class LLM_Model:
    def __init__(self, llm_model):
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        self.message = []
        self.response = []
        self.response_token =[]
        self.input_token_all = 0
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client
        )
        self.llm_config = {
            "model": llm_model,
            "max_tokens": 2000,
            "temperature": 0.0,
            "seed": 2026,
        }

    
    def infer(self, messages):
        response = self.openai_client.chat.completions.create(**self.llm_config, messages=messages)
        

        message_content = response.choices[0].message.content

        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
        

        reasoning_tokens = 0
        input_tokens = 0
        total_tokens = 0
        
        if hasattr(response, 'usage') and response.usage is not None:
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', 0)
            
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details is not None:
                reasoning_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
        

        self.message.append(messages[1]['content'])
        self.response.append(reasoning_content)
        self.response_token.append(reasoning_tokens)
        

        self.input_token_all+=total_tokens
        

        return message_content



def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val
