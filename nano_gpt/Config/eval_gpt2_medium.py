EVAL_GPT2_MEDIUM_CONFIG = {
    'n_layer': 24,
    'n_head': 16,
    'n_embd': 1024,
    'batch_size': 8,
    'eval_iters': 500,  # use more iterations to get good estimate
    'eval_only': True,
    'init_from': 'gpt2-medium',
}
