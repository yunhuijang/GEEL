python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 272 \
--wandb_on online \
--string_type bfs \
--lr 0.0005 \
--num_samples 1000 \
--batch_size 128 \
--tree_pos \
--pos_type emb \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 272 \
--wandb_on online \
--string_type bfs \
--lr 0.0005 \
--num_samples 1000 \
--batch_size 128 \
--tree_pos \
--pos_type pad