import os
import pickle

import numpy as np
import requests

from nano_gpt.Method.path import createFileFolder

dataset_folder_path = '/home/chli/chLi/nanoGPT/shakespeare_char/'

# download the tiny shakespeare dataset
input_file_path = f'{dataset_folder_path}input.txt'
createFileFolder(input_file_path)

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/' + \
        'karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = dict(enumerate(chars))


def encode(string):
    # encoder: take a string, output a list of integers
    return [stoi[c] for c in string]


def decode(value):
    # decoder: take a list of integers, output a string
    return ''.join([itos[i] for i in value])


# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(f'{dataset_folder_path}train.bin')
val_ids.tofile(f'{dataset_folder_path}val.bin')

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(f'{dataset_folder_path}meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
