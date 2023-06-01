# nano GPT

## Source

```bash
https://github.com/karpathy/nanoGPT
```

## Download

```bash
https://openwebtext2.readthedocs.io/en/latest/
-->
OpenWebText
```

## Install

```bash
conda create -n nanogpt python=3.8
conda activate nanogpt
./setup.sh
```

## Run

```bash
python demo.py
```

## Train

```bash
python train.py
```

## Source Script

Download mini dataset

```bash
python data/shakespeare_char/prepare.py
```

Train on mini dataset

```bash
python train.py config/train_shakespeare_char.py
```

Test trained model

```bash
python sample.py --out_dir=out-shakespeare-char
```

Finetune pretrained model

```bash
python train.py config/finetune_shakespeare.py
```

Inference

```bash
python sample.py \
  --init_from=gpt2-xl \
  --start="What is the answer to life, the universe, and everything?" \
  --num_samples=5 --max_new_tokens=100
```

## Enjoy it~
