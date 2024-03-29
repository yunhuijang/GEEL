python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name point \
--max_epochs 1000 \
--check_sample_every_n_epoch 2 \
--replicate 0 \
--max_len 15923 \
--wandb_on disabled \
--string_type adj_list \
--batch_size 16 \
--num_samples 30 \
--lr 0.0001 \
--sample_batch_size 5