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
from moses.metrics.metrics import get_all_metrics

from data.dataset import EgoDataset, ComDataset, EnzDataset, GridDataset, GridSmallDataset, QM9Dataset, ZINCDataset, PlanarDataset, SBMDataset, ProteinsDataset
from data.data_utils import dfs_string_to_tree, tree_to_adj, check_validity, bfs_string_to_tree, adj_to_graph, check_tree_validity, generate_final_tree_red, fix_symmetry, generate_initial_tree_red
from data.mol_utils import adj_to_graph_mol, mols_to_smiles, check_adj_validity_mol, mols_to_nx, fix_symmetry_mol, canonicalize_smiles
from evaluation.evaluation import compute_sequence_accuracy, compute_sequence_cross_entropy, save_graph_list, load_eval_settings, eval_graph_list
from plot import plot_graphs_list
from model.lstm_generator import LSTMGenerator
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
        self.k = hparams.k
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
            with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.order}/{hparams.dataset_name}' + f'_smiles_train.txt', 'r') as f:
                self.train_smiles = f.readlines()
                self.train_smiles = canonicalize_smiles(self.train_smiles)
            with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.order}/{hparams.dataset_name}' + f'_smiles_test.txt', 'r') as f:
                self.test_smiles = f.readlines()
                self.test_smiles = canonicalize_smiles(self.test_smiles)
        with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.order}/{hparams.dataset_name}' + f'_test_graphs.pkl', 'rb') as f:
            self.test_graphs = pickle.load(f)
        self.train_dataset, self.val_dataset, self.test_dataset = [dataset_cls(split, self.string_type, self.order, is_tree=hparams.tree_pos, k=self.k)
                                                                   for split in ['train', 'val', 'test']]
        self.max_depth = hparams.max_depth

    def setup_model(self, hparams):
        self.model = LSTMGenerator(
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

    def validation_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            self.check_samples()

    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        string_list, org_string_list = self.sample(num_samples)
        
        if not self.trainer.sanity_checking:
            
            if self.hparams.string_type == 'dfs':
                valid_string_list = [string for string in string_list if check_validity(string)]
                sampled_trees = [dfs_string_to_tree(string) for string in valid_string_list]
                valid_sampled_trees = [tree for tree in sampled_trees if (tree.depth() <= self.max_depth) and check_tree_validity(tree)]
            elif self.hparams.string_type in ['bfs', 'group', 'bfs-deg', 'bfs-deg-group', 'qm9', 'zinc']:
                valid_string_list = [string for string in string_list if len(string)>0 and len(string)%4 == 0]
                is_zinc = False
                if self.hparams.string_type in ['zinc', 'zinc-red']:
                    is_zinc = True
                sampled_trees = [bfs_string_to_tree(string, is_zinc, self.k) 
                                for string in tqdm(valid_string_list, "Sampling: converting string to tree")]
                valid_sampled_trees = [tree for tree in sampled_trees if (tree.depth() <= self.max_depth) and check_tree_validity(tree)]
            
            elif self.hparams.string_type in ['bfs-red', 'group-red', 'group-red-3', 'qm9-red', 'zinc-red']:
                valid_string_list = [string for string in string_list if len(string)>0]
                sampled_trees = [generate_initial_tree_red(string, self.k) for string in valid_string_list]
                valid_sampled_trees = [tree for tree in sampled_trees if check_tree_validity(tree)]
                valid_sampled_trees = [generate_final_tree_red(tree, self.k) for tree in tqdm(valid_sampled_trees, "Sampling: converting string to tree")]
                
            wandb.log({"validity": len(valid_string_list)/len(string_list)})
            # write down string

            
            if self.hparams.string_type in ['zinc', 'qm9', 'zinc-red', 'qm9-red']:
                # valid_sampled_trees = sampled_trees[:len(self.test_graphs)]
                adjs = [fix_symmetry_mol(tree_to_adj(tree)).numpy() for tree in tqdm(valid_sampled_trees, "Sampling: converting tree into adj")]
                valid_adjs = [valid_adj for valid_adj in [check_adj_validity_mol(adj) for adj in adjs] if valid_adj is not None]
                mols_no_correct = [adj_to_graph_mol(adj) for adj in valid_adjs]
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
                with open(f'samples/smiles/{save_dir}.txt', 'w') as f:
                    for smiles in gen_smiles:
                        f.write(f'{smiles}\n')
                scores = get_all_metrics(gen=gen_smiles, device=self.device, n_jobs=8, test=self.test_smiles, train=self.train_smiles, k=len(gen_smiles))
                scores_nspdk = eval_graph_list(self.test_graphs, mols_to_nx(mols), methods=['nspdk'])['nspdk']
                metrics_dict = scores
                metrics_dict['unique'] = scores[f'unique@{len(gen_smiles)}']
                del metrics_dict[f'unique@{len(gen_smiles)}']
                metrics_dict['NSPDK'] = scores_nspdk
                metrics_dict['validity_wo_cor'] = sum(no_corrects) / num_mols
                wandb.log(metrics_dict)
            else:
                table = wandb.Table(columns=['Orginal', 'String', 'Validity'])
                if 'red' in self.hparams.string_type:
                    org_string_list = [''.join(org) for org in org_string_list]
                    string_list = [''.join(s) for s in string_list]
                for org_string, string in zip(org_string_list, string_list):
                    table.add_data(org_string, string, (len(string)>0 and len(string)%4 == 0))
                wandb.log({'strings': table})
                if len(sampled_trees) > 0:
                    tree_validity = len(valid_sampled_trees) / len(sampled_trees)
                else:
                    tree_validity = 0
                wandb.log({"tree-validity": tree_validity})
                # valid_sampled_trees = valid_sampled_trees[:len(self.test_graphs)]
                adjs = [fix_symmetry(tree_to_adj(tree, self.k)).numpy() for tree in tqdm(valid_sampled_trees, "Sampling: converting tree into graph")]
                sampled_graphs = [adj_to_graph(adj) for adj in adjs]
                save_graph_list(self.hparams.dataset_name, self.ts, sampled_graphs, valid_string_list, string_list, org_string_list)
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
        offset = 0
        string_list = []
        org_string_list = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                sequences = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)

            strings = [untokenize(sequence, self.hparams.string_type, self.hparams.k)[0] for sequence in sequences.tolist()]
            org_strings = [untokenize(sequence, self.hparams.string_type, self.hparams.k)[1] for sequence in sequences.tolist()]
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