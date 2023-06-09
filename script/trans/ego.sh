python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 66 \
<<<<<<< HEAD
--wandb_on disabled \
--string_type adj_list \
=======
--wandb_on online \
--string_type adj_list_ego \
>>>>>>> c8c3b2bc210f85a73db2f99f718eb03cbd366630
--lr 0.001 \
--batch_size 128 \
--num_samples 200 \