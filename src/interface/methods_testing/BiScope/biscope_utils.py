import os
import json
import time
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

# Minimal model zoo mapping model keys to pretrained model names.

MODEL_ZOO = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gemma-2b': 'google/gemma-1.1-2b-it',
    'gemma-7b': 'google/gemma-1.1-7b-it', 
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'CodeLlama' :'meta-llama/CodeLlama-7b-hf'          #<------------------ ADDED BY ME
}


def generate(model, tokenizer, input_ids, trigger_length, target_length):
    """
    Generate additional tokens using the model's generation API.
    
    Parameters:
      model: the language model for generation.
      tokenizer: associated tokenizer.
      input_ids: input token IDs (either 1D or 2D).
      trigger_length: the length of the prompt (number of tokens to skip in the output).
      target_length: the number of new tokens to generate.
      
    Returns:
      Generated tokens (as a 2D tensor) after removing the trigger tokens.
    """
    config = model.generation_config
    config.max_new_tokens = target_length
    # If input_ids is 1D, add a batch dimension; otherwise, assume it's already 2D.
    if input_ids.dim() == 1:
        input_ids = input_ids.to(model.device).unsqueeze(0)
    else:
        input_ids = input_ids.to(model.device)
    # Create an attention mask of the same shape.
    attn_masks = torch.ones(input_ids.shape, device=input_ids.device)
    # Generate new tokens.
    out = model.generate(
        input_ids, 
        attention_mask=attn_masks,
        generation_config=config,
        pad_token_id=tokenizer.pad_token_id
    )[0]
    # Return output tokens after the prompt (slice along dimension 1).
    return out[trigger_length:]


def compute_fce_loss(logits, targets, text_slice):
    """
    Compute the FCE loss by shifting indices by 1.
    Returns a NumPy array of loss values.
    """
    loss = CrossEntropyLoss(reduction='none')(
        logits[0, text_slice.start-1:text_slice.stop-1, :],
        targets
    )
    return loss.detach().cpu().numpy()

def compute_bce_loss(logits, targets, text_slice):
    """
    Compute the BCE loss without shifting indices.
    Returns a NumPy array of loss values.
    """
    loss = CrossEntropyLoss(reduction='none')(
        logits[0, text_slice, :],
        targets
    )
    return loss.detach().cpu().numpy()
