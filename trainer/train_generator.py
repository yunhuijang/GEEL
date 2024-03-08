import argparse
import torch
from tqdm import tqdm
import time
import numpy as np
import pytorch_lightning as pl
import wandb
from time import gmtime, strftime
import os
import networkx as nx

from data.dataset import EgoDataset, ComDataset, EnzDataset, GridDataset, GridSmallDataset, QM9Dataset, ZINCDataset, PlanarDataset, SBMDataset, ProteinsDataset, LobsterDataset, PointCloudDataset, EgoLargeDataset, MosesDataset, GuacamolDataset
from data.dataset import Grid10000Dataset, Grid1000Dataset, Grid20000Dataset, Grid2000Dataset, Grid5000Dataset, Grid500Dataset
from data.data_utils import adj_to_graph, load_graphs, map_samples_to_adjs, get_max_len
from data.mol_utils import canonicalize_smiles
from evaluation.evaluation import compute_sequence_cross_entropy, save_graph_list, load_eval_settings, eval_graph_list, evaluate_molecules
from plot import plot_graphs_list
from data.tokens import untokenize
from data.mol_tokens import untokenize_mol
from data.mol_utils import map_featured_samples_to_adjs
from evaluation.evaluation_spectre import eval_fraction_unique_non_isomorphic_valid, eval_fraction_isomorphic, eval_fraction_unique, is_planar_graph, eval_acc_planar_graph, eval_acc_grid_graph, eval_acc_sbm_graph, is_sbm_graph, eval_acc_lobster_graph, eval_fraction_isomorphic_ego, eval_fraction_unique_ego, eval_fraction_unique_non_isomorphic_valid_ego, is_grid_graph, is_lobster_graph


DATA_DIR = "resource"

class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.ts = strftime('%b%d-%H:%M:%S', gmtime())
        
    def setup_datasets(self, hparams):
        self.string_type = hparams.string_type
        self.order = hparams.order
        self.is_token = hparams.is_token
        self.vocab_size = hparams.vocab_size
        self.dataset_name = hparams.dataset_name
        self.replicate = hparams.replicate
        self.max_epochs = hparams.max_epochs
        self.is_random_order = hparams.is_random_order
        dataset_cls = {
            "GDSS_grid": GridDataset,
            "GDSS_ego": EgoDataset,
            "GDSS_com": ComDataset,
            "GDSS_enz": EnzDataset,
            "grid_small": GridSmallDataset,
            'planar': PlanarDataset,
            'sbm': SBMDataset,
            'proteins': ProteinsDataset,
            'lobster': LobsterDataset,
            'point': PointCloudDataset,
            'ego': EgoLargeDataset,
            'qm9': QM9Dataset,
            'zinc': ZINCDataset,
            'moses': MosesDataset,
            'guacamol': GuacamolDataset,
            'grid-500': Grid500Dataset,
            'grid-1000': Grid1000Dataset,
            'grid-2000': Grid2000Dataset,
            'grid-5000': Grid5000Dataset, 
            'grid-10000': Grid10000Dataset,
            'grid-20000': Grid20000Dataset
        }.get(hparams.dataset_name)
        if self.is_random_order:
            _, self.val_graphs, self.test_graphs = load_graphs(hparams.dataset_name, self.order, seed=self.replicate)
            
            train_datasets = []
            
            for epoch in tqdm(range(self.max_epochs), 'Order training graphs'):
                file_path = f'ordered_dataset/{hparams.dataset_name}/{epoch%hparams.random_order_epoch}.pt'
                if os.path.isfile(file_path):
                    train_graphs = torch.load(file_path)
                else:
                    train_graphs = load_graphs(hparams.dataset_name, self.order, epoch, is_train=True)
                self.train_graphs = train_graphs
                train_dataset = dataset_cls(train_graphs, self.string_type, self.is_token, self.vocab_size, self.order)
                train_datasets.append(train_dataset)
            self.train_dataset = train_datasets
            self.num_nodes = get_max_len([self.train_graphs, self.val_graphs, self.test_graphs])[1]
            train_bw = max([dataset.bw for dataset in train_datasets])
            self.val_dataset, self.test_dataset = [dataset_cls(graphs, self.string_type, self.is_token, self.vocab_size, self.order)
                                                                    for graphs in [self.val_graphs, self.test_graphs]]
            self.bw = max(train_bw, self.val_dataset.bw, self.test_dataset.bw)
            
        else:
            self.train_graphs, self.val_graphs, self.test_graphs = load_graphs(hparams.dataset_name, self.order, hparams.replicate)
            graphs_list = [self.test_graphs, self.val_graphs, self.test_graphs]
            self.num_nodes = get_max_len(graphs_list)[1]
            self.train_dataset, self.val_dataset, self.test_dataset = [dataset_cls(graphs, self.string_type, self.is_token, self.vocab_size, self.order)
                                                                        for graphs in [self.train_graphs, self.val_graphs, self.test_graphs]]
            self.bw = max(self.train_dataset.bw, self.val_dataset.bw, self.test_dataset.bw)
        
        if self.dataset_name in ['qm9', 'zinc', 'moses', 'guacamol']:
            with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.dataset_name}' + f'_smiles_train.txt', 'r') as f:
                self.train_smiles = f.readlines()
                self.train_smiles = canonicalize_smiles(self.train_smiles)
            with open(f'{DATA_DIR}/{hparams.dataset_name}/{hparams.dataset_name}' + f'_smiles_test.txt', 'r') as f:
                self.test_smiles = f.readlines()
                self.test_smiles = canonicalize_smiles(self.test_smiles)
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            )
        
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data, self.hparams.dataset_name, self.hparams.string_type, self.hparams.order, self.hparams.is_token, self.hparams.vocab_size)
        statistics["loss/total"] = loss
        # statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data, ignore_index=0)[0]

        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)
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
        adj_lists, org_string_list, generation_time = self.sample(num_samples)
        wandb.log({"generation_time": round(generation_time, 3)})
        
        if not self.trainer.sanity_checking:
            if self.dataset_name in ['qm9', 'zinc', 'moses', 'guacamol']:
            # self.string_type in ['adj_seq_merge', 'adj_seq_rel_merge']:
                weighted_adjs, xs = map_featured_samples_to_adjs(adj_lists, self.string_type)
                if len(weighted_adjs) > 0:
                    evaluate_molecules(weighted_adjs, xs, self.dataset_name, self.test_graphs, self.device, self.test_smiles, self.train_smiles)
            else:
                adjs = map_samples_to_adjs(adj_lists, self.string_type, self.is_token)
                wandb.log({'ratio': len(adjs) / len(adj_lists)})
                
                sampled_graphs = [adj_to_graph(adj) for adj in adjs]
                save_graph_list(self.hparams.dataset_name, wandb.run.id + "_" + str(self.current_epoch), sampled_graphs)
                plot_dir = f'{self.hparams.dataset_name}/{wandb.run.id}'
                plot_graphs_list(sampled_graphs, title=self.current_epoch, save_dir=plot_dir)
                print(f'current: {os.getcwd()}')
                wandb.log({"samples": wandb.Image(f'samples/fig/{plot_dir}/{self.current_epoch}.png')})

                # GDSS evaluation
                methods, kernels = load_eval_settings('')
                if len(sampled_graphs) == 0:
                    mmd_results = {'degree': np.nan, 'orbit': np.nan, 'cluster': np.nan, 'spectral': np.nan}
                else:
                    mmd_results = eval_graph_list(self.test_graphs, sampled_graphs[:len(self.test_graphs)], methods=methods, kernels=kernels)
                    for graph in self.test_graphs:
                        nx.set_node_attributes(graph, 0, "label")
                        nx.set_edge_attributes(graph, 1, "label")
                    for graph in sampled_graphs:
                        nx.set_node_attributes(graph, 0, "label")
                        nx.set_edge_attributes(graph, 1, "label")
                    scores_nspdk = eval_graph_list(self.test_graphs, sampled_graphs[:len(self.test_graphs)], methods=['nspdk'])['nspdk']
                    mmd_results['nspdk'] = scores_nspdk
                    mmd_results['avg_mmd'] = (mmd_results['degree'] + mmd_results['orbit'] + mmd_results['cluster'])/3
                wandb.log(mmd_results)
                
                # SPECTRE evaluation
                gen_graphs = sampled_graphs[:len(self.test_graphs)]
                if len(gen_graphs) > 0:
                    if self.hparams.dataset_name == 'lobster':
                        spectre_valid = eval_acc_lobster_graph(gen_graphs)
                        _, spectre_un, spectre_vun = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.train_graphs, is_lobster_graph)
                    elif self.hparams.dataset_name == 'planar':
                        spectre_valid = eval_acc_planar_graph(gen_graphs)
                        _, spectre_un, spectre_vun = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.train_graphs, is_planar_graph)
                    elif self.hparams.dataset_name == 'GDSS_grid':
                        spectre_valid = eval_acc_grid_graph(gen_graphs)
                        _, spectre_un, spectre_vun = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.train_graphs, is_grid_graph)
                    elif self.hparams.dataset_name == 'ego':
                        spectre_valid = 0
                        _, spectre_un, spectre_vun = eval_fraction_unique_non_isomorphic_valid_ego(gen_graphs, self.train_graphs)
                    else:
                        spectre_valid = 0
                        _, spectre_un, spectre_vun = eval_fraction_unique_non_isomorphic_valid(gen_graphs, self.train_graphs)
                    if self.hparams.dataset_name == 'ego':
                        spectre_unique = eval_fraction_unique_ego(gen_graphs)
                        spectre_novel = eval_fraction_isomorphic_ego(gen_graphs, self.train_graphs)
                    else:
                        spectre_unique = eval_fraction_unique(gen_graphs)
                        spectre_novel = round(1.0-eval_fraction_isomorphic(gen_graphs, self.train_graphs),3)
                    spectre_results = {'spec_valid': spectre_valid, 'spec_unique': spectre_unique, 'spec_novel': spectre_novel,
                                    'spec_un': spectre_un, 'spec_vun': spectre_vun}
                    wandb.log(spectre_results)
                
    def sample(self, num_samples):
        '''
        generate graphs
        '''
        offset = 0
        string_list = []
        org_string_list = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            self.model.eval()
            with torch.no_grad():
                t0 = time.perf_counter()
                sequences = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)
                generation_time = time.perf_counter() - t0
                
            if (self.string_type in ['adj_list_diff_ni']) and (self.dataset_name in ['qm9', 'zinc']):
                strings = [untokenize_mol(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.vocab_size)[0] for sequence in sequences.tolist()]
                org_strings = [untokenize_mol(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.vocab_size)[1] for sequence in sequences.tolist()]
            else:
                strings = [untokenize(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.order, self.vocab_size)[0] for sequence in sequences.tolist()]
                org_strings = [untokenize(sequence, self.hparams.dataset_name, self.string_type, self.is_token, self.order, self.vocab_size)[1] for sequence in sequences.tolist()]
            string_list.extend(strings)
            org_string_list.extend(org_strings)
            
        return string_list, org_string_list, generation_time
        
    @staticmethod
    def add_args(parser):
        

        return parser