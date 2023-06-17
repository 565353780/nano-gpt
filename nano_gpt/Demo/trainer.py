#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nano_gpt.Module.trainer import Trainer


def demo():
    config_name = 'train_shakespeare_char'
    model_file_path = './output/model_best.pt'

    trainer = Trainer(config_name, model_file_path)
    trainer.train()
    return True
