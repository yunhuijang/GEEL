python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 36 \
--wandb_on online \
--string_type adj_flatten \
--lr 0.0001 \
--batch_size 128 \
--num_samples 200 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 36 \
--wandb_on online \
--string_type adj_flatten \
--lr 0.0002 \
--batch_size 128 \
--num_samples 200 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 36 \
--wandb_on online \
--string_type adj_flatten \
--lr 0.0005 \
--batch_size 128 \
--num_samples 200 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 15 \
--wandb_on online \
--string_type adj_flatten \
--lr 0.001 \
--batch_size 128 \
--num_samples 200 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 15 \
--wandb_on online \
--string_type adj_flatten_sym \
--lr 0.0001 \
--batch_size 128 \
--num_samples 200 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 36 \
--wandb_on online \
--string_type adj_flatten_sym \
--lr 0.0002 \
--batch_size 128 \
--num_samples 200 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 15 \
--wandb_on online \
--string_type adj_flatten_sym \
--lr 0.0005 \
--batch_size 128 \
--num_samples 200 \
--is_token \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 15 \
--wandb_on online \
--string_type adj_flatten_sym \
--lr 0.001 \
--batch_size 128 \
--num_samples 200 \
--is_token \