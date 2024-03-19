python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name planar \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 181 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.0001 \
--batch_size 128 \
--num_samples 200 \
--is_random_order \
--random_order_epoch 1000 \
--pe no \
;
python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name planar \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 181 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.0002 \
--batch_size 128 \
--num_samples 200 \
--is_random_order \
--random_order_epoch 1000 \
--pe no \
;
python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name planar \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 181 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.0005 \
--batch_size 128 \
--num_samples 200 \
--is_random_order \
--random_order_epoch 1000 \
--pe no \
;
python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name planar \
--max_epochs 5000 \
--check_sample_every_n_epoch 200 \
--replicate 0 \
--max_len 181 \
--wandb_on online \
--string_type adj_list_diff_ni_rel \
--lr 0.001 \
--batch_size 128 \
--num_samples 200 \
--is_random_order \
--random_order_epoch 1000 \
--pe no