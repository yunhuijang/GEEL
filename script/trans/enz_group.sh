python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_enz \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 1720 \
--wandb_on online \
--string_type bfs-deg-group \
--lr 0.002 \
--batch_size 32 \
--sample_batch_size 50 \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_enz \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 1720 \
--wandb_on online \
--string_type bfs-deg-group \
--lr 0.001 \
--batch_size 32 \
--sample_batch_size 50