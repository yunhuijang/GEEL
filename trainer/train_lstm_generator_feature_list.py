import argparse
import os
from os import listdir
from os.path import isfile, join
import torch

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks import ModelCheckpoint, Timer

from model.lstm_generator_feature_list import LSTMGeneratorFeatureList
from trainer.train_generator import BaseGeneratorLightningModule

from signal import signal, SIGPIPE, SIG_DFL   
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)  
signal(SIGPIPE,SIG_DFL)

os.environ["WANDB__SERVICE_WAIT"] = "300"


class LSTMGeneratorFeatureListLightningModule(BaseGeneratorLightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)

    def setup_model(self, hparams):
        self.model = LSTMGeneratorFeatureList(
            emb_size=hparams.emb_size,
            dropout=hparams.dropout,
            string_type=hparams.string_type,
            dataset=hparams.dataset_name,
            num_layers=hparams.num_layers,
            is_token=hparams.is_token,
            vocab_size=hparams.vocab_size,
            max_len=hparams.max_len,
            bw=self.bw,
            num_nodes=self.num_nodes,
            pe=hparams.pe,
            is_simple_token=hparams.is_simple_token,
            order=hparams.order
        )
        
    ### 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_args(parser):
       
        parser.add_argument("--dataset_name", type=str, default="qm9")
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--order", type=str, default="C-M")
        parser.add_argument("--replicate", type=int, default=0)
        #
        parser.add_argument("--emb_size", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=0.0001)
        
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=50)
        parser.add_argument("--num_samples", type=int, default=100)
        parser.add_argument("--sample_batch_size", type=int, default=100)
        parser.add_argument("--max_epochs", type=int, default=2000)
        parser.add_argument("--wandb_on", type=str, default='disabled')
        
        parser.add_argument("--group", type=str, default='lstm')
        parser.add_argument("--model", type=str, default='lstm')
        parser.add_argument("--max_len", type=int, default=116)
        parser.add_argument("--string_type", type=str, default='adj_list_diff_ni')
        
        
        # transformer
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--is_token", action="store_true")
        parser.add_argument("--pe", type=str, default='node')
        parser.add_argument("--vocab_size", type=int, default=400)
        
        parser.add_argument("--run_id", type=str, default=None)
        parser.add_argument("--is_simple_token", action="store_true")
        parser.add_argument("--is_random_order", action="store_true")

        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    LSTMGeneratorFeatureListLightningModule.add_args(parser)
    hparams = parser.parse_args()
    if hparams.run_id == None:
        wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}-{hparams.string_type}', 
                               project='alt', group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
        model = LSTMGeneratorFeatureListLightningModule(hparams)
        ckpt_path=None
    else:
       # for resume
        wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}-{hparams.string_type}', 
                               project='alt', group=f'{hparams.group}', mode=f'{hparams.wandb_on}',
                               version=hparams.run_id, resume="must")
        model = LSTMGeneratorFeatureListLightningModule(hparams)
        
        ckpt_path = f"resource/checkpoint/{hparams.dataset_name}/{hparams.run_id}/last"
        file_list = [f for f in listdir(ckpt_path) if isfile(join(ckpt_path, f))]
        ckpt_path += f'/{file_list[0]}'
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        
    
    wandb.config.update(hparams, allow_val_change=True)

    
    checkpoint_callback_val = ModelCheckpoint(
        dirpath=os.path.join("resource/checkpoint/", hparams.dataset_name, wandb.run.id, 'val'),
        monitor='val/loss/total',
    )
    checkpoint_callback_train = ModelCheckpoint(
        dirpath=os.path.join("resource/checkpoint/", hparams.dataset_name, wandb.run.id, 'train'),
        monitor='train/loss/total', save_on_train_epoch_end=True
    )
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=os.path.join("resource/checkpoint/", hparams.dataset_name, wandb.run.id, 'last')
    )

    timer = Timer(duration="14:00:00:00")
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        default_root_dir="/resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback_val, checkpoint_callback_train, checkpoint_callback_last, timer],
        logger=wandb_logger,
        resume_from_checkpoint=ckpt_path
    )
    trainer.fit(model, ckpt_path=ckpt_path)
    wandb.log({"train_time": round(timer.time_elapsed("train"),3)})    