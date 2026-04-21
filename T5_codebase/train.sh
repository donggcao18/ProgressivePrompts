#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

export CUDA_VISIBLE_DEVICES=0,1
port=$(shuf -i25000-30000 -n1)

# CONCODE CodeTrans CodeSearchNet BFP KodCode RunBugRun TheVault_Csharp CoST
accelerate launch --config_file accelerate_config.yaml train_t5_cl.py \
    --save_dir log \
    --save_name codet5p-770m \
    --task_list CONCODE CodeTrans CodeSearchNet BFP KodCode RunBugRun TheVault_Csharp CoST \
    --model_name Salesforce/codet5p-770m \
    --num_epochs 3 \
    --batch_size 8 \
    --seq_len 512 \
    --lr 0.3 \
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
    --max_train 100000 \
    --max_eval 1000 \
    --max_test 5000 \
    --prefix_len 50 \
    --start_task 4 \
    --prefix_path /data/scratch/projects/punim1928/HUST/east/CodeGR/Dense/ProgressivePrompts/T5_codebase/log/codet5p-770m_20260421_061517/prompts_after_task3_BFP.npy


