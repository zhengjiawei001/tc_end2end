import os

import numpy as np
import onnxruntime
import psutil
import torch
from flask import Flask, request
from tqdm import trange
from transformers import BertTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)


def softmax(array):
    array -= array.max(axis=-1, keepdims=True)
    exp_array = np.exp(array)
    probs = exp_array / exp_array.sum(axis=1, keepdims=True)
    return np.mean(probs, axis=0)[1]


def init_model(export_model_path_0, export_model_path_1, export_model_path_2):
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

    session_0 = onnxruntime.InferenceSession(export_model_path_0, sess_options)
    session_1 = onnxruntime.InferenceSession(export_model_path_1, sess_options)
    session_2 = onnxruntime.InferenceSession(export_model_path_2, sess_options)
    return session_0, session_1, session_2


def infer(query_a, query_b):
    inputs = tokenizer([query_a, query_b], [query_b, query_a], return_tensors='pt',
                       add_special_tokens=True, truncation='longest_first', max_length=30)
    inputs = {key: value.numpy() for key, value in inputs.items()}
    logits_0 = session_0.run(None, inputs)[0]
    logits_1 = session_1.run(None, inputs)[0]
    logits_2 = session_2.run(None, inputs)[0]
    logits = np.concatenate([logits_0, logits_1,
                             logits_2], axis=0)
    prob = softmax(logits)
    return prob


@app.route("/tccapi", methods=['GET', 'POST'])
def tccapi():
    data = request.get_json()
    if (data == b"exit"):
        print("received exit command, exit now")
        os._exit(0)
    input_list = request.form.getlist("input")
    index_list = request.form.getlist("index")
    response_batch = {}
    response_batch["results"] = []
    for i in range(len(index_list)):
        index_str = index_list[i]
        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split("\t")
            query_A = elems[0].strip()
            query_B = elems[1].strip()
            predict = infer(query_A, query_B)
            response["predict"] = float(predict)
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
        response_batch["results"].append(response)

    return response_batch


if __name__ == '__main__':
    model_path_0 = './user_data/nezha-results-0.onnx'
    model_path_1 = './user_data/nezha-results-1.onnx'
    model_path_2 = './user_data/nezha-results-2.onnx'
    vocab_path = './user_data/vocab.txt'
    # query_A = '29 13 67 12 68 69 70 16'
    # query_B = '71 10 72 29 68 69 70'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    session_0, session_1, session_2 = init_model(model_path_0, model_path_1, model_path_2)
    
    #infer('12 5 239 243 29 1001 126 1405 11', '29 485 12 251 1405 11')
    app.run(host="0.0.0.0", port=8080)
