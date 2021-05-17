import argparse
import os

import onnx
import torch
from transformers import BertTokenizer

from transformers import BertForSequenceClassification
from src.nezha.model import NeZhaForSequenceClassification_

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def export_onnx():
    model = model_class.from_pretrained(model_path, torchscript=True)
    model.eval().to(device)
    inputs = tokenizer('32,34,44','44,32,12',padding='max_length', truncation='longest_first',
                       max_length=30, return_tensors="pt")
    dummy_inputs = (
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
        inputs["token_type_ids"].to(device),
    )
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    output_names = ['logits']

    opset_version = 11
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,
                          args=dummy_inputs,
                          f=export_model_path,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes={'input_ids': symbolic_names,
                                        'attention_mask': symbolic_names,
                                        'token_type_ids': symbolic_names},
                          verbose=True)
        print("Model onnx path ", export_model_path)
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print('over')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=1, type=int)

    args = parser.parse_args()

    model_path = f'./user_data/finetune-nezha-results/checkpoint-{args.id}'
    vocab_path = './user_data/vocab.txt'
    export_model_path = f'./user_data/nezha-results-{args.id}.onnx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    model_class = NeZhaForSequenceClassification_
    export_onnx()
