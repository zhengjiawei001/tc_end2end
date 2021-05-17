import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import AdamW
from transformers import (
    BertTokenizer,
)

from src.nezha.model import NeZhaForSequenceClassification
from src.utils import WarmupLinearSchedule, Lookahead, load_data, load_skf_data, seed_random


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.1, alpha=0.3, emb_name='embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


def train(config, train_dataloader):
    model = NeZhaForSequenceClassification.from_pretrained(config['model_path'])

    lr_scheduler, optimizer = train_pre(config, model, train_dataloader)
    model.to(config['device'])
    pgd = PGD(model)
    epoch_iterator = trange(config['epochs'])
    global_steps = 0
    train_loss = 0.

    # if config['n_gpus'] > 1:
    #     model = nn.DataParallel(model)

    train_step(config, epoch_iterator, pgd, global_steps, lr_scheduler, model, optimizer, train_dataloader, train_loss)

    model_save_path = os.path.join(config['output_path'], f'checkpoint-{config["model_path"][-1]}')
    model_save = model.module if hasattr(model, 'module') else model
    model_save.save_pretrained(model_save_path)


def train_step(config, epoch_iterator, pgd, global_steps, lr_scheduler, model, optimizer, train_dataloader, train_loss):
    K=3
    for _ in epoch_iterator:
        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            input_batch = {item: value.to(config['device']) for item, value in list(batch.items())}
            loss = model(**input_batch)[0]
            loss.backward()
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = model(**input_batch)[0]
                loss_adv.backward()
            pgd.restore()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            global_steps += 1

            train_iterator.set_postfix_str(f'loss: {loss.item():.4f}')


def train_pre(config, model, train_dataloader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'],
                      correct_bias=False, eps=1e-8)
    optimizer = Lookahead(optimizer, 5, 1)
    total_steps = config['epochs'] * len(train_dataloader)
    lr_scheduler = WarmupLinearSchedule(optimizer,
                                        warmup_steps=int(config['warmup_ratio'] * total_steps),
                                        t_total=total_steps)
    return lr_scheduler, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='best_model_ckpt_0', type=str)
    parser.add_argument('--seed', default=2020, type=int)
    args = parser.parse_args()

    config = {
        'data_cache_path': './user_data/data.pkl',
        'output_path': './user_data/finetune-nezha-results',
        'vocab_path': './user_data/vocab.txt',
        'model_path': f'./user_data/pretrained-nezha-base/{args.model_path}',
        'max_seq_len': 30,
        'learning_rate': 2e-5,
        'eps': 0.1,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'batch_size': 128,
        'epochs': 4,
        'logging_step': 5000,
        'seed': args.seed,
        'device': 'cuda'
    }

    # if not torch.cuda.is_available():
    #     config['device'] = 'cpu'
    # else:
    #     config['n_gpus'] = torch.cuda.device_count()
    #     config['batch_size'] *= config['n_gpus']

    if not os.path.exists(config['output_path']):
        os.makedirs((config['output_path']))

    tokenizer = BertTokenizer.from_pretrained(config['vocab_path'])
    collate_fn, train_dev_data = load_data(config, tokenizer)

    train_dataloader = load_skf_data(collate_fn, config, train_dev_data)

    seed_random(config['seed'])

    train(config, train_dataloader)


if __name__ == '__main__':
    main()
