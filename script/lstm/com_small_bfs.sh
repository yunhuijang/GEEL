python trainer/train_lstm_generator.py \
--order C-M \
--dataset_name GDSS_com \
--max_epochs 1000 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 304 \
--wandb_on online \
--string_type bfs