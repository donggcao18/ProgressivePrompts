#!/bin/bash

# Run run_t5.py with all default values
python3 train_t5_cl.py \
    --save_dir data \
    --save_name t5_continual \
    --task_list CONCODE,CodeTrans,CodeSearchNet,BFP \
    --model_name Salesforce/codet5-770m \
    --num_epochs 2 \
    --multitask 0 \
    --batch_size 32 \
    --seq_len 512 \
    --prefix_len 10 \
    --prefix_path "" \
    --lr 0.1 \
    --memory_perc 0.0 \
    --data_replay_freq -1 \
    --select_k_per_class -1 \
    --test_eval_after_every_task 1 \
    --progressive 1 \
    --freeze_weights 1 \
    --freeze_except xxxxxxx \
    --get_test_subset 0 \
    --early_stopping 1 \
    --prefix_MLP None \
    --mlp_layer_norm 1 \
    --bottleneck_size 800
