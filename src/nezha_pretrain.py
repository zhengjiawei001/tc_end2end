# -*- coding: utf-8 -*-
import argparse
import os

from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    logging
)

from src.utils import build_path, seed_random, read_data, TcDataset, TcCollator
from src.nezha.model import NeZhaForMaskedLM


def main():
    logging.set_verbosity_info()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='best_model_ckpt_0', type=str)
    parser.add_argument('--seed', default=202105, type=int)
    args = parser.parse_args()
    seed_random(args.seed)
    data_path = './user_data/duality_pair_pretrain_no_nsp.txt'
    vocab_path = './user_data/vocab.txt'
    model_path = './user_data/nezha-cn-base'
    output_path = './user_data/pretrained-nezha-base'

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    data = read_data(data_path, tokenizer)


    train_dataset = TcDataset(data)

    model = NeZhaForMaskedLM.from_pretrained(model_path)
    model.resize_token_embeddings(tokenizer.vocab_size)

    data_collator = TcCollator(max_seq_len=30, tokenizer=tokenizer, mlm_probability=0.15)

    logging_path = os.path.join(output_path, 'log')
    model_save_path = os.path.join(output_path, args.model_path)
    tokenizer_and_config = os.path.join(output_path, 'tokenizer_and_config')
    build_path(model_save_path)
    build_path(logging_path)
    build_path(tokenizer_and_config)

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        learning_rate=6e-5,
        num_train_epochs=130,
        per_device_train_batch_size=128,
        logging_steps=5000,
        fp16=True,
        fp16_backend='amp',
        load_best_model_at_end=True,
        prediction_loss_only=True,
        logging_dir=logging_path,
        logging_first_step=True,
        dataloader_num_workers=4,
        seed=2021
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(tokenizer_and_config)


if __name__ == '__main__':
    main()
