#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

export CUDA_VISIBLE_DEVICES=1
port=$(shuf -i25000-30000 -n1)

python train_t5_cl.py \
    --save_dir data \
    --save_name codet5p-770m \
    --task_list CONCODE CodeTrans CodeSearchNet BFP KodCode RunBugRun TheVault_Csharp CoST \
    --model_name Salesforce/codet5p-770m \
    --num_epochs 5 \
    --batch_size 16 \
    --seq_len 512 \
    --lr 0.1 \
    --progressive 1 \
    --freeze_weights 1 \
    --freeze_except xxxxxxx \
    --prefix_MLP None \
    --mlp_layer_norm 1 \
    --bottleneck_size 800 \
    --early_stopping 1 \
    --get_test_subset 0 \
    --select_k_per_class -1 \
    --memory_perc 0.0 \
    --data_replay_freq -1 \
    --multitask 0 \
    --test_eval_after_every_task 1 \
    --max_train 100 \
    --max_eval 10 \
    --max_test 50 \