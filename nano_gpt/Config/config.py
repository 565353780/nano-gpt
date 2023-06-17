from dataclasses import dataclass

from nano_gpt.Config.train_shakespeare_char import \
    TRAIN_SHAKESPEARE_CHAR_CONFIG
from nano_gpt.Config.finetune_shakespeare import \
    FINETUNE_SHAKESPEARE_CONFIG
from nano_gpt.Config.train_gpt2 import TRAIN_GPT2_CONFIG
from nano_gpt.Config.eval_gpt2 import EVAL_GPT2_CONFIG
from nano_gpt.Config.eval_gpt2_medium import \
    EVAL_GPT2_MEDIUM_CONFIG
from nano_gpt.Config.eval_gpt2_large import EVAL_GPT2_LARGE_CONFIG
from nano_gpt.Config.eval_gpt2_xl import EVAL_GPT2_XL_CONFIG

CONFIG_MAP = {
    'train_shakespeare_char': TRAIN_SHAKESPEARE_CHAR_CONFIG,
    'finetune_shakespeare': FINETUNE_SHAKESPEARE_CONFIG,
    'train_gpt2': TRAIN_GPT2_CONFIG,
    'eval_gpt2': EVAL_GPT2_CONFIG,
    'eval_gpt2_medium': EVAL_GPT2_MEDIUM_CONFIG,
    'eval_gpt2_large': EVAL_GPT2_LARGE_CONFIG,
    'eval_gpt2_xl': EVAL_GPT2_XL_CONFIG,
}


@dataclass
class GPTConfig:
    block_size: int = 1024
    # GPT-2 vocab_size of 50257,
    # padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2.
    # False: a bit better and faster
    bias: bool = True
