from typing import *

BACKEND = 'flash_attn' 
DEBUG = False

def __from_env():
    import os
    import torch
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    elif torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] < 8:
                # Pre-Ampere GPUs (T4 sm_75, V100 sm_70) do not support FlashAttention
                BACKEND = 'sdpa'
        except Exception:
            pass
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
