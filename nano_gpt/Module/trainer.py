#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import pickle
import time
from contextlib import nullcontext

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group

from nano_gpt.Config.config import GPTConfig
from nano_gpt.Model.nano import GPT
from nano_gpt.Method.time import getCurrentTime
from nano_gpt.Method.path import createFileFolder


class Trainer(object):
    def __init__(self):
        # -----------------------------------------------------------------------------
        # default config values designed to train a gpt2 (124M) on OpenWebText
        # I/O
        self.out_dir = './output/'
        self.eval_interval = 2000
        self.log_interval = 1
        self.eval_iters = 200
        self.eval_only = False
        self.always_save_checkpoint = True
        # 'scratch' or 'resume' or 'gpt2*'
        self.init_from = 'scratch'
        # wandb logging
        self.wandb_log = False  # disabled by default
        self.wandb_project = 'owt'
        self.wandb_run_name = 'gpt2'  # 'run' + str(time.time())
        # data
        self.dataset = 'openwebtext'
        # used to simulate larger batch sizes
        self.gradient_accumulation_steps = 5 * 8
        # if gradient_accumulation_steps > 1, this is the micro-batch size
        self.batch_size = 12
        self.block_size = 1024
        # model
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        # for pretraining 0 is good, for finetuning try 0.1+
        self.dropout = 0.0
        self.bias = False
        # adamw optimizer
        self.learning_rate = 6e-4
        self.max_iters = 600000
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        # learning rate decay settings
        self.decay_lr = True
        self.warmup_iters = 2000
        # should be ~= max_iters per Chinchilla
        self.lr_decay_iters = 600000
        # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        self.min_lr = 6e-5
        # DDP settings
        self.backend = 'nccl'
        # system
        # examples: 'cpu', 'cuda', 'cuda:0'
        self.device = 'cuda'
        # 'float32', 'bfloat16', or 'float16'
        # the latter will auto implement a GradScaler
        self.dtype = 'bfloat16'
        self.compile = True
        # -----------------------------------------------------------------------------

        # config
        self.config = None

        # ddp
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        self.master_process = True
        self.seed_offset = 0
        self.ddp_local_rank = None

        # ctx
        self.ctx = nullcontext()
        self.device_type = None

        # model
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.model_args = dict(n_layer=self.n_layer, n_head=self.n_head,
                               n_embd=self.n_embd, block_size=self.block_size,
                               bias=self.bias, vocab_size=None,
                               dropout=self.dropout)
        self.checkpoint = None
        self.model = None
        self.unoptimized_model = None

        # scaler
        self.scaler = None

        # logger
        self.step = 0
        self.eval_step = 0
        self.loss_min = float('inf')
        self.eval_loss_min = float('inf')
        self.log_folder_name = getCurrentTime()
        self.summary_writer = None

        self.loadConfig()
        self.loadDDP()
        self.loadCTX()
        self.loadDataset()
        self.loadModel()
        self.loadScaler()
        self.loadOptimizer()
        self.compileModel()
        return

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter("./logs/" + self.log_folder_name +
                                            "/")
        return True

    def loadConfig(self):
        config_keys = [k for k, v in globals().items() if not k.startswith(
            '_') and isinstance(v, (int, float, bool, str))]
        # overrides from command line or config file
        exec(open('nano_gpt/Config/configurator.py').read())
        self.config = {k: globals()[k]
                       for k in config_keys}  # will be useful for logging
        # -----------------------------------------------------------------------------
        return True

    def loadDDP(self):
        if not self.ddp:
            tokens_per_iter = self.gradient_accumulation_steps * \
                self.batch_size * self.block_size
            print(f"tokens per iteration will be: {tokens_per_iter:,}")
            return True

        init_process_group(backend=self.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        self.device = f'cuda:{self.ddp_local_rank}'
        torch.cuda.set_device(self.device)
        # this process will do logging, checkpointing etc.
        self.master_process = ddp_rank == 0
        self.seed_offset = ddp_rank
        assert self.gradient_accumulation_steps % \
            torch.cuda.device_count() == 0
        self.gradient_accumulation_steps //= torch.cuda.device_count()
        tokens_per_iter = self.gradient_accumulation_steps * \
            ddp_world_size * self.batch_size * self.block_size
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        return True

    def loadCTX(self):
        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)
        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # for later use in torch.autocast
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32,
                   'bfloat16': torch.bfloat16,
                   'float16': torch.float16}[self.dtype]
        if self.device_type != 'cpu':
            self.ctx = torch.amp.autocast(
                device_type=self.device_type, dtype=ptdtype)
        return True

    def loadDataset(self):
        # poor man's data loader
        self.data_dir = os.path.join('data', self.dataset)
        self.train_data = np.memmap(os.path.join(
            self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(self.data_dir, 'val.bin'),
                                  dtype=np.uint16, mode='r')
        return True

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(
            (data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            (data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        if self.device_type != 'cuda':
            return x.to(self.device), y.to(self.device)

        # pin arrays x,y, move them to GPU asynchronously by non_blocking=True
        return x.pin_memory().to(self.device, non_blocking=True), \
            y.pin_memory().to(self.device, non_blocking=True)

    def resumeModel(self):
        print(f"Resuming training from {self.out_dir}")

        # resume training from a checkpoint.
        ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
        self.checkpoint = torch.load(ckpt_path, map_location=self.device)
        checkpoint_model_args = self.checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd',
                  'block_size', 'bias', 'vocab_size']:
            self.model_args[k] = checkpoint_model_args[k]

        # create the model
        gptconf = GPTConfig(**self.model_args)
        self.model = GPT(gptconf)
        state_dict = self.checkpoint['model']

        # fix the keys of the state dictionary :(
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.iter_num = self.checkpoint['iter_num']
        self.best_val_loss = self.checkpoint['best_val_loss']

        if 'step' in self.checkpoint:
            self.step = self.checkpoint['step']
        if 'eval_step' in self.checkpoint:
            self.eval_step = self.checkpoint['eval_step']
        if 'loss_min' in self.checkpoint:
            self.loss_min = self.checkpoint['loss_min']
        if 'eval_loss_min' in self.checkpoint:
            self.eval_loss_min = self.checkpoint['eval_loss_min']
        if 'log_folder_name' in self.checkpoint:
            self.log_folder_name = self.checkpoint['log_folder_name']
        return True

    def loadModel(self):
        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        if self.init_from == 'scratch':
            print("Initializing a new model from scratch")

            if meta_vocab_size is None:
                print(
                    "defaulting to vocab_size of GPT-2 to 50304 \
                    (50257 rounded up for efficiency)")
                self.model_args['vocab_size'] = 50304
            else:
                self.model_args['vocab_size'] = meta_vocab_size

            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)

        elif self.init_from == 'resume':
            self.resumeModel()

        elif self.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {self.init_from}")

            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=self.dropout)
            self.model = GPT.from_pretrained(
                self.init_from, override_args)
            for k in ['n_layer', 'n_head', 'n_embd',
                      'block_size', 'bias', 'vocab_size']:
                self.model_args[k] = getattr(self.model.config, k)

        # crop down the model block size if desired, using model surgery
        if self.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.block_size)
            # so that the checkpoint will have the right value
            self.model_args['block_size'] = self.block_size

        self.model.to(self.device)

        self.loadSummaryWriter()
        return True

    def loadScaler(self):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.dtype == 'float16'))
        return True

    def loadOptimizer(self):
        self.optimizer = self.model.configure_optimizers(
            self.weight_decay, self.learning_rate,
            (self.beta1, self.beta2), self.device_type)
        if self.init_from == 'resume':
            assert self.checkpoint is not None
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        self.checkpoint = None
        return True

    def compileModel(self):
        if self.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)
        return True

    def warpModel(self):
        if not self.ddp:
            return True

        self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        return True

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / \
            (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        # coeff ranges 0..1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def updateLR(self):
        lr = self.get_lr(
            self.iter_num) if self.decay_lr else self.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def saveModel(self, save_model_file_path):
        raw_model = self.model.module if self.ddp else self.model

        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model_args,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'step': self.step,
            'eval_step': self.eval_step,
            'loss_min': self.loss_min,
            'eval_loss_min': self.eval_loss_min,
            'log_folder_name': self.log_folder_name,
        }

        createFileFolder(save_model_file_path)

        torch.save(checkpoint, save_model_file_path)
        return True

    def trainStep(self):
        return True

    def train(self):
        X, Y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0

        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0
        while True:
            lr = self.updateLR()

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()

                print(
                    f"step {self.iter_num}: train loss {losses['train']:.4f}, \
                    val loss {losses['val']:.4f}")

                self.summary_writer.add_scalar(
                    'Train/iter', self.iter_num, self.step)
                self.summary_writer.add_scalar(
                    'Train/loss', losses['train'], self.step)
                self.summary_writer.add_scalar(
                    'Eval/loss', losses['val'], self.step)
                self.summary_writer.add_scalar(
                    'Param/lr', lr, self.step)
                self.summary_writer.add_scalar(
                    'Param/mfu', running_mfu*100, self.step)

                if losses['val'] < self.best_val_loss or \
                        self.always_save_checkpoint:
                    self.best_val_loss = losses['val']
                    if self.iter_num > 0:
                        self.saveModel(self.out_dir + 'ckpt.pt')
            if self.iter_num == 0 and self.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    # scale the loss to account for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(
                        batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break

        if ddp:
            destroy_process_group()
        return True
