python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name ego \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 1071 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.0001 \
--batch_size 16 \
--num_samples 300 \
--sample_batch_size 20 \
--is_random_order \
--random_order_epoch 1000 \
--pe learn \
;
python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name ego \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 1071 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.0002 \
--batch_size 16 \
--num_samples 300 \
--sample_batch_size 20 \
--is_random_order \
--random_order_epoch 1000 \
--pe learn \
;
python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name ego \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 1071 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.0005 \
--batch_size 16 \
--num_samples 300 \
--sample_batch_size 20 \
--is_random_order \
--random_order_epoch 1000 \
--pe learn \
;
python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name ego \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 1071 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.001 \
--batch_size 16 \
--num_samples 300 \
--sample_batch_size 20 \
--is_random_order \
--random_order_epoch 1000 \
--pe learn