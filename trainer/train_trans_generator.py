import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES']="1"

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from datetime import date

from evaluation.evaluation import compute_sequence_cross_entropy
from model.trans_generator import TransGenerator
from trainer.train_generator import BaseGeneratorLightningModule

from signal import signal, SIGPIPE, SIG_DFL   
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)  
signal(SIGPIPE,SIG_DFL)


class TransGeneratorLightningModule(BaseGeneratorLightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)

    def setup_model(self, hparams):
        self.model = TransGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            input_dropout=hparams.input_dropout,
            dropout=hparams.dropout,
            max_len=hparams.max_len,
            string_type=hparams.string_type,
            learn_pos=hparams.learn_pos,
            abs_pos=hparams.abs_pos,
            data_name=hparams.dataset_name,
            bw=self.bw,
            num_nodes=self.num_nodes,
            is_token=hparams.is_token,
            vocab_size=hparams.vocab_size
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

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data, self.hparams.dataset_name, self.hparams.string_type, self.hparams.is_token, self.hparams.vocab_size)
        statistics["loss/total"] = loss
        # statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data, ignore_index=0)[0]

        return loss, statistics

    
    @staticmethod
    def add_args(parser):
       
        parser.add_argument("--dataset_name", type=str, default="lobster")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--order", type=str, default="C-M")
        parser.add_argument("--replicate", type=int, default=0)
        #
        parser.add_argument("--emb_size", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=0.0002)
        
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=2)
        parser.add_argument("--num_samples", type=int, default=100)
        parser.add_argument("--sample_batch_size", type=int, default=100)
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--wandb_on", type=str, default='disabled')
        
        parser.add_argument("--group", type=str, default='string')
        parser.add_argument("--model", type=str, default='trans')
        parser.add_argument("--max_len", type=int, default=99)
        parser.add_argument("--string_type", type=str, default='adj_list')
        
        
        # transformer
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=512)
        parser.add_argument("--input_dropout", type=float, default=0.0)
        parser.add_argument("--gradient_clip_val", type=float, default=1.0)
        parser.add_argument("--learn_pos", action="store_true")
        parser.add_argument("--abs_pos", action="store_true")
        parser.add_argument("--is_token", action="store_true")
        parser.add_argument("--vocab_size", type=int, default=400)
        

        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    TransGeneratorLightningModule.add_args(parser)
    hparams = parser.parse_args()
    
    wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}-{hparams.string_type}', 
                               project='alt', group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
    
    wandb.config.update(hparams)
    

    model = TransGeneratorLightningModule(hparams)
    checkpoint_callback_val = ModelCheckpoint(
        dirpath=os.path.join("resource/checkpoint/", hparams.dataset_name, date.today().isoformat(), wandb.run.id, 'val'),
        monitor='val/loss/total',
    )
    checkpoint_callback_train = ModelCheckpoint(
        dirpath=os.path.join("resource/checkpoint/", hparams.dataset_name, date.today().isoformat(), wandb.run.id, 'train'),
        monitor='train/loss/total', save_on_train_epoch_end=True
    )
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=os.path.join("resource/checkpoint/", hparams.dataset_name, date.today().isoformat(), wandb.run.id, 'last')
    )
    timer = Timer(duration="14:00:00:00")
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        default_root_dir="/resource/log/",
        max_epochs=hparams.max_epochs,
        gradient_clip_val=hparams.gradient_clip_val,
        callbacks=[checkpoint_callback_last, checkpoint_callback_val, checkpoint_callback_train, timer],
        logger=wandb_logger
    )
    trainer.fit(model)
    wandb.log({"train_time": round(timer.time_elapsed("train"),3)})    