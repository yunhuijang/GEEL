import argparse
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from datetime import date
import torch
import time
from moses.metrics.metrics import get_all_metrics

os.environ["WANDB__SERVICE_WAIT"] = "300"

from evaluation.evaluation import compute_sequence_cross_entropy, compute_sequence_cross_entropy_feature, eval_graph_list
from model.trans_generator_feature import TransGeneratorFeature
from model.trans_generator import TransGenerator
from trainer.train_generator import BaseGeneratorLightningModule
from data.tokens import untokenize
from data.mol_tokens import untokenize_mol
from data.data_utils import map_samples_to_adjs
from data.mol_utils import mols_to_smiles, mols_to_nx, map_featured_samples_to_adjs, adj_x_to_graph_mol

from signal import signal, SIGPIPE, SIG_DFL   
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)  
signal(SIGPIPE,SIG_DFL)


class TransGeneratorFeatureLightningModule(BaseGeneratorLightningModule):
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
        self.model_feature = TransGeneratorFeature(
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
            num_nodes=self.num_nodes
        )

    ### 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda sequences: (pad_sequence([sequence[0] for sequence in sequences], batch_first=True, padding_value=0),
                                          pad_sequence([sequence[1] for sequence in sequences], batch_first=True, padding_value=0)),
            num_workers=self.hparams.num_workers,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda sequences: (pad_sequence([sequence[0] for sequence in sequences], batch_first=True, padding_value=0),
                                          pad_sequence([sequence[1] for sequence in sequences], batch_first=True, padding_value=0)),
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda sequences: (pad_sequence([sequence[0] for sequence in sequences], batch_first=True, padding_value=0),
                                          pad_sequence([sequence[1] for sequence in sequences], batch_first=True, padding_value=0)),
            num_workers=self.hparams.num_workers,
        )

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        logits = self.model(batched_data)
        logits_feature = self.model_feature(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data[0], self.hparams.dataset_name, self.hparams.string_type, self.hparams.is_token, self.hparams.vocab_size)
        loss_feature = compute_sequence_cross_entropy_feature(logits_feature, batched_data[1], self.hparams.dataset_name, self.hparams.string_type)
        total_loss = loss + loss_feature
        statistics["loss/total"] = total_loss
        statistics["loss/adj"] = loss
        statistics["loss/feature"] = loss_feature
        # statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data, ignore_index=0)[0]

        return total_loss, statistics

    def sample(self, num_samples):
        '''
        generate graphs
        '''
        offset = 0
        string_list = []
        org_string_list = []
        string_list_feature = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            self.model.eval()
            self.model_feature.eval()
            with torch.no_grad():
                t0 = time.perf_counter()
                sequences = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)
                sequences_feature = self.model_feature.decode(cur_num_samples, sequences, max_len=self.hparams.max_len, device=self.device)
                generation_time = time.perf_counter() - t0
            
            # for adj    
            strings = [untokenize(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.vocab_size)[0] for sequence in sequences.tolist()]
            org_strings = [untokenize(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.vocab_size)[1] for sequence in sequences.tolist()]
            string_list.extend(strings)
            org_string_list.extend(org_strings)
            
            # for feature
            strings_feature = [untokenize_mol(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.vocab_size)[0] for sequence in sequences_feature.tolist()]
            # org_strings = [untokenize(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.vocab_size)[1] for sequence in sequences.tolist()]
            string_list_feature.extend(strings_feature)
            # org_string_list.extend(org_strings)
            
        return string_list, org_string_list, string_list_feature, generation_time
    
    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        adj_lists, org_string_list, feature_lists, generation_time = self.sample(num_samples)
        wandb.log({"generation_time": round(generation_time, 3)})
        
        if not self.trainer.sanity_checking:

            weighted_adjs, xs = map_featured_samples_to_adjs(adj_lists, feature_lists, self.string_type)
            mols_no_correct = [adj_x_to_graph_mol(weighted_adj, x) for weighted_adj, x in zip(weighted_adjs, xs) if len(weighted_adj) > 1]
            mols_no_correct = [elem for elem in mols_no_correct if elem[0] is not None]
            mols = [elem[0] for elem in mols_no_correct]
            no_corrects = [elem[1] for elem in mols_no_correct]
            num_mols = len(mols)
            gen_smiles = mols_to_smiles(mols)
            gen_smiles = [smi for smi in gen_smiles if len(smi)]
            table = wandb.Table(columns=['SMILES'])
            for s in gen_smiles:
                table.add_data(s)
            wandb.log({'SMILES': table})
            save_dir = f'{self.hparams.dataset_name}/{self.ts}'
            scores_nspdk = eval_graph_list(self.test_graphs, mols_to_nx(mols), methods=['nspdk'])['nspdk']
            with open(f'samples/smiles/{save_dir}.txt', 'w') as f:
                for smiles in gen_smiles:
                    f.write(f'{smiles}\n')
            scores = get_all_metrics(gen=gen_smiles, device=self.device, n_jobs=8, test=self.test_smiles, train=self.train_smiles, k=len(gen_smiles))
            
            metrics_dict = scores
            metrics_dict['unique'] = scores[f'unique@{len(gen_smiles)}']
            del metrics_dict[f'unique@{len(gen_smiles)}']
            metrics_dict['NSPDK'] = scores_nspdk
            metrics_dict['validity_wo_cor'] = sum(no_corrects) / num_mols
            wandb.log(metrics_dict)
    
    @staticmethod
    def add_args(parser):
       
        parser.add_argument("--dataset_name", type=str, default="qm9")
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
        parser.add_argument("--string_type", type=str, default='adj_seq_rel')
        
        
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
        
        # parser.add_argument("--is_joint_adj", action="store_true")
        parser.add_argument("--run_id", type=str, default=None)
        

        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    TransGeneratorFeatureLightningModule.add_args(parser)
    hparams = parser.parse_args()
    if hparams.run_id == None:
        wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}-{hparams.string_type}', 
                               project='alt', group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
        model = TransGeneratorFeatureLightningModule(hparams)
        ckpt_path=None
    else:
       # for resume
        wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}-{hparams.string_type}', 
                               project='alt', group=f'{hparams.group}', mode=f'{hparams.wandb_on}',
                               version=hparams.run_id, resume="must")
        model = TransGeneratorFeatureLightningModule(hparams)
        
        ckpt_path = f"resource/checkpoint/{hparams.dataset_name}/{hparams.run_id}/last"
        file_list = [f for f in listdir(ckpt_path) if isfile(join(ckpt_path, f))]
        ckpt_path += f'/{file_list[0]}'
        ckpt = torch.load(ckpt_path)
        model = model.load_from_checkpoint(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    wandb.config.update(hparams)
    
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
        gradient_clip_val=hparams.gradient_clip_val,
        callbacks=[checkpoint_callback_last, checkpoint_callback_val, checkpoint_callback_train, timer],
        logger=wandb_logger,
        resume_from_checkpoint=ckpt_path
    )
    trainer.fit(model, ckpt_path=ckpt_path)
    wandb.log({"train_time": round(timer.time_elapsed("train"),3)})    