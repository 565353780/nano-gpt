EVAL_GPT2_XL_CONFIG = {
    'n_layer': 48,
    'n_head': 25,
    'n_embd': 1600,
    'batch_size': 8,
    'eval_iters': 500,  # use more iterations to get good estimate
    'eval_only': True,
    'init_from': 'gpt2-xl',
}
