#!/usr/bin/env bash


pushd .
cd ../

export BERT_BASE_DIR=./data/cased_L-12_H-768_A-12
export GLUE_DIR=./data/glue_data
export TRAINED_MODDEL_DIR=./model/xnli/
export CUDA_VISIBLE_DEVICES=3

python run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/XNLI \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$TRAINED_MODDEL_DIR

popd