#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nano_gpt.Module.trainer import Trainer


def demo():
    model_file_path = './output/model_best.pt'
    trainer = Trainer(model_file_path)
    trainer.train()
    return True
