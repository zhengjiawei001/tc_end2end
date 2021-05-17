python -m src.preprocess

echo 'start pretrain'
CUDA_VISIBLE_DEVICES=0 nohup python -m src.nezha_pretrain  --model_path best_model_ckpt_0  --seed 2020 &> ./user_data/nezha_prerain_log0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m src.nezha_pretrain  --model_path best_model_ckpt_1  --seed 2021 &> ./user_data/nezha_prerain_log1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m src.nezha_pretrain  --model_path best_model_ckpt_2  --seed 2022 &> ./user_data/nezha_prerain_log2.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python -m src.bert_wwm_pretrain_1 &> ./user_data/bert_wwm_pretrain-log1.txt &

python -m src.transition --status pretrain

echo 'start training'
CUDA_VISIBLE_DEVICES=0 nohup python -m src.nezha_train --model_path best_model_ckpt_0 --seed 2021 &> ./user_data/nezha_finetune_log0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m src.nezha_train --model_path best_model_ckpt_1 --seed 202105 &> ./user_data/nezha_finetune_log1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m src.nezha_train --model_path best_model_ckpt_2 --seed 2021050 &> ./user_data/nezha_finetune_log2.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python -m src.train_wwm --model_dir best_model_ckpt_0 --seed 20210503 &> ./user_data/wwm-finetune-log1.txt &

python -m src.transition --status finetune
echo 'finetune finish'
python -m src.model_onnx  --id 0
python -m src.model_onnx  --id 1
python -m src.model_onnx  --id 2
#
#
python -m src.info_onnx

echo "Over ..."


