python trainer/train_trans_generator.py \
--order C-M \
--dataset_name sbm \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 1129 \
--wandb_on online \
--string_type adj_list \
--lr 0.0002 \
--batch_size 8 \
--num_samples 60 \
--sample_batch_size 20 \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name sbm \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 1129 \
--wandb_on online \
--string_type adj_list \
--lr 0.0001 \
--batch_size 8 \
--num_samples 60 \
--sample_batch_size 20