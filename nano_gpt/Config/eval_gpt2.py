EVAL_GPT2_CONFIG = {
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'batch_size': 8,
    'eval_iters': 500,  # use more iterations to get good estimate
    'eval_only': True,
    'init_from': 'gpt2',
}
