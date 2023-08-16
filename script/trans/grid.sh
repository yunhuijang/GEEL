python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 136 \
--wandb_on online \
--string_type adj_seq \
--lr 0.0001 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 50 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 136 \
--wandb_on online \
--string_type adj_seq \
--lr 0.0002 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 50 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 136 \
--wandb_on online \
--string_type adj_seq \
--lr 0.0005 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 50 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 136 \
--wandb_on online \
--string_type adj_seq \
--lr 0.001 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 20 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 94 \
--wandb_on online \
--string_type adj_seq_rel \
--lr 0.0001 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 50 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 94 \
--wandb_on online \
--string_type adj_seq_rel \
--lr 0.0002 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 50 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 94 \
--wandb_on online \
--string_type adj_seq_rel \
--lr 0.0005 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 50 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_grid \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 94 \
--wandb_on online \
--string_type adj_seq_rel \
--lr 0.001 \
--batch_size 64 \
--num_samples 100 \
--sample_batch_size 50 \
--is_token