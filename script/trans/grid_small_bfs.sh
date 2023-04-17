python trainer/train_trans_generator.py \
--order C-M \
--dataset_name grid_small \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 660 \
--wandb_on online \
--string_type bfs \
--num_samples 1000 \
--lr 0.0001 \
--tree_pos \
--pos_type emb \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name grid_small \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 660 \
--wandb_on online \
--string_type bfs \
--num_samples 1000 \
--lr 0.0001 \
--tree_pos \
--pos_type emb \