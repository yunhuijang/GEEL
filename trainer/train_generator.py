import argparse
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from time import gmtime, strftime
import pickle
import os
#from moses.metrics.metrics import get_all_metrics

from model.trans_generator import TransGenerator
from data.dataset import EgoDataset, ComDataset, EnzDataset, GridDataset, GridSmallDataset, QM9Dataset, ZINCDataset, PlanarDataset, SBMDataset, ProteinsDataset
from data.data_utils import adj_to_graph, fix_symmetry, adj_list_to_adj
#from data.mol_utils import adj_to_graph_mol, mols_to_smiles, check_adj_validity_mol, mols_to_nx, fix_symmetry_mol, canonicalize_smiles
from evaluation.evaluation import compute_sequence_accuracy, compute_sequence_cross_entropy, save_graph_list, load_eval_settings, eval_graph_list
from plot import plot_graphs_list
from data.tokens import untokenize

DATA_DIR = "resource"

class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.ts = strftime('%b%d-%H:%M:%S', gmtime())
        wandb.config['ts'] = self.ts
        
    def setup_datasets(self, hparams):
        self.string_type = hparams.string_type
        self.order = hparams.order
        #self.k = hparams.k
    
        dataset_cls = {
            "GDSS_grid": GridDataset,
            "GDSS_ego": EgoDataset,
            "GDSS_com": ComDataset,
            "GDSS_enz": EnzDataset,
            "grid_small": GridSmallDataset,
            'qm9': QM9Dataset,
            'zinc': ZINCDataset,
            'planar': PlanarDataset,
            'sbm': SBMDataset,
            'proteins': ProteinsDataset
        }.get(hparams.dataset_name)
        if hparams.dataset_name in ['qm9', 'zinc']:
            
        #     with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.order}/{hparams.dataset_name}' + f'_smiles_train.txt', 'r') as f:
        #         self.train_smiles = f.readlines()
        #         self.train_smiles = canonicalize_smiles(self.train_smiles)
        #     with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.order}/{hparams.dataset_name}' + f'_smiles_test.txt', 'r') as f:
        #         self.test_smiles = f.readlines()
        #         self.test_smiles = canonicalize_smiles(self.test_smiles)
        # with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.order}/{hparams.dataset_name}' + f'_test_graphs.pkl', 'rb') as f:
        #     self.test_graphs = pickle.load(f)


            with open(f'{DATA_DIR}/{hparams.dataset_name}' + f'_smiles_train.txt', 'r') as f:
                self.train_smiles = f.readlines()
                self.train_smiles = canonicalize_smiles(self.train_smiles)
            with open(f'{DATA_DIR}/{hparams.dataset_name}' + f'_smiles_test.txt', 'r') as f:
                self.test_smiles = f.readlines()
                self.test_smiles = canonicalize_smiles(self.test_smiles)
        with open(f'{DATA_DIR}/{hparams.dataset_name}' + f'_test_graphs.pkl', 'rb') as f:
            self.test_graphs = pickle.load(f)
        self.train_dataset, self.val_dataset, self.test_dataset = [dataset_cls(split, self.order)
                                                                   for split in ['train', 'val', 'test']]

        self.max_depth = hparams.max_depth

    def setup_model(self, hparams):
        self.model = TransGenerator(
            emb_size=hparams.emb_size,
            dropout=hparams.dropout,
            dataset=hparams.dataset_name
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            )
        
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        # decoding
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data, ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data, ignore_index=0)[0]

        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            # self.log(f"train/{key}", val, on_step=True, logger=True)
            wandb.log({f"train/{key}": val})
        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            wandb.log({f"val/{key}": val})
            self.log(f"val/{key}", val, on_step=False, on_epoch=True, logger=True)
        pass

    # def validation_epoch_end(self, outputs):
    #     if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
    #         self.check_samples()

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            self.check_samples()

    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        adj_lists, org_string_list = self.sample(num_samples)
        
        if not self.trainer.sanity_checking:
            
            adjs = [torch.tensor(adj_list_to_adj(adj_list)) for adj_list in adj_lists if len(adj_list) > 0]
            wandb.log({'ratio': len([adj_list for adj_list in adj_lists if len(adj_list) > 0]) / len(adj_lists)})
            #adjs = [torch.tensor(adj_list_to_adj(adj_list)) for adj_list in adj_lists]
            sampled_graphs = [adj_to_graph(adj) for adj in adjs]
            save_graph_list(self.hparams.dataset_name, self.ts, sampled_graphs)
            plot_dir = f'{self.hparams.dataset_name}/{self.ts}'
            plot_graphs_list(sampled_graphs, save_dir=plot_dir)
            wandb.log({"samples": wandb.Image(f'./samples/fig/{plot_dir}/title.png')})

            # GDSS evaluation
            methods, kernels = load_eval_settings('')
            if len(sampled_graphs) == 0:
                mmd_results = {'degree': np.nan, 'orbit': np.nan, 'cluster': np.nan, 'spectral': np.nan}
            else:
                mmd_results = eval_graph_list(self.test_graphs, sampled_graphs[:len(self.test_graphs)], methods=methods, kernels=kernels)
            wandb.log(mmd_results)

    def sample(self, num_samples):
        '''
        generate graphs
        '''
        # TODO: maybe need change afterwards
        offset = 0
        string_list = []
        org_string_list = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                sequences = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)

            strings = [untokenize(sequence, self.hparams.string_type)[0] for sequence in sequences.tolist()]
            org_strings = [untokenize(sequence, self.hparams.string_type)[1] for sequence in sequences.tolist()]
            string_list.extend(strings)
            org_string_list.extend(org_strings)
            
        return string_list, org_string_list
        
    @staticmethod
    def add_args(parser):
        

        return parser


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)


    hparams = parser.parse_args()
    wandb_logger = WandbLogger(name=f'{hparams.dataset_name}', project='k2g', 
                               group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
    wandb.config.update(hparams)
    
    model = BaseGeneratorLightningModule(hparams)
    wandb.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        logger=wandb_logger
    )
    trainer.fit(model)