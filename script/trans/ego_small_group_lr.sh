python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 272 \
--wandb_on online \
--string_type group \
--lr 0.002 \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 272 \
--wandb_on online \
--string_type group \
--lr 0.001 \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 272 \
--wandb_on online \
--string_type group \
--lr 0.0005 \
;
python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_ego \
--max_epochs 500 \
--check_sample_every_n_epoch 20 \
--replicate 0 \
--max_len 272 \
--wandb_on online \
--string_type group \
--lr 0.0002