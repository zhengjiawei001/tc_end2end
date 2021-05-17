import argparse
import os


def start_finetune():
    is_start = (os.path.exists(nezha_model_file0) and os.path.exists(nezha_model_file1)
                and os.path.exists(nezha_model_file2))
    return is_start


def waiting():
    while True:
        is_start = start_finetune()
        if is_start:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', default='pretrain', type=str)
    args = parser.parse_args()
    if args.status == 'pretrain':
        pretrained_model_dir = './user_data/pretrained-nezha-base'
        nezha_model_file0 = os.path.join(pretrained_model_dir, 'best_model_ckpt_0/pytorch_model.bin')
        nezha_model_file1 = os.path.join(pretrained_model_dir, 'best_model_ckpt_1/pytorch_model.bin')
        nezha_model_file2 = os.path.join(pretrained_model_dir, 'best_model_ckpt_2/pytorch_model.bin')
    else:
        finetune_model_dir = './user_data/finetune-nezha-results'
        nezha_model_file0 = os.path.join(finetune_model_dir, 'checkpoint-0/pytorch_model.bin')
        nezha_model_file1 = os.path.join(finetune_model_dir, 'checkpoint-1/pytorch_model.bin')
        nezha_model_file2 = os.path.join(finetune_model_dir, 'checkpoint-2/pytorch_model.bin')
    waiting()
