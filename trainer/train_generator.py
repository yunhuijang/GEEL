import argparse
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from time import gmtime, strftime
import pickle

from data.dataset import EgoDataset, ComDataset, EnzDataset, GridDataset, GridSmallDataset
from data.data_utils import dfs_string_to_tree, tree_to_adj, check_validity, train_val_test_split, bfs_string_to_tree, adj_to_graph
from evaluation.evaluation import compute_sequence_accuracy, compute_sequence_cross_entropy, save_graph_list, load_eval_settings, eval_graph_list
from plot import plot_graphs_list
from model.lstm_generator import LSTMGenerator
from data.tokens import untokenize


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
        dataset_cls = {
            "GDSS_grid": GridDataset,
            "GDSS_ego": EgoDataset,
            "GDSS_com": ComDataset,
            "GDSS_enz": EnzDataset,
            "grid_small": GridSmallDataset
        }.get(hparams.dataset_name)
        try:
            with open(f'resource/{hparams.dataset_name}/{hparams.dataset_name}' + f'_test_graphs.pkl', 'rb') as f:
                self.test_graphs = pickle.load(f)
        except:
            with open(f'gcg/resource/{hparams.dataset_name}/{hparams.dataset_name}' + f'_test_graphs.pkl', 'rb') as f:
                self.test_graphs = pickle.load(f)
        self.train_dataset, self.val_dataset, self.test_dataset = [dataset_cls(split, self.string_type, is_tree=(not hparams.tree_pos))
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
        # loss, statistics = self.shared_step(batched_data)
        # for key, val in statistics.items():
        #     self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)
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
            elif self.hparams.string_type in ['bfs', 'group', 'bfs-deg', 'bfs-deg-group']:
                valid_string_list = [string for string in string_list if len(string)>0 and len(string)%4 == 0]
                sampled_trees = [bfs_string_to_tree(string) 
                                for string in tqdm(valid_string_list, "Sampling: converting string to tree")]
                
                
            wandb.log({"validity": len(valid_string_list)/len(string_list)})
            # write down string
            table = wandb.Table(columns=['Orginal', 'String', 'Validity'])
            for org_string, string in zip(org_string_list, string_list):
                table.add_data(org_string, string, (len(string)>0 and len(string)%4 == 0))
            wandb.log({'strings': table})
            valid_sampled_trees = [tree for tree in sampled_trees if tree.depth() <= self.max_depth]
            valid_sampled_trees = valid_sampled_trees[:len(self.test_graphs)]
            sampled_graphs = [adj_to_graph(tree_to_adj(tree).numpy()) 
                              for tree in tqdm(valid_sampled_trees, "Sampling: converting tree into graph")]
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
                if self.hparams.string_type in ['group', 'bfs-deg-group']:
                    sequences = self.model.decode(cur_num_samples, max_len=int(self.hparams.max_len/4), device=self.device)
                else:
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
    wandb_logger = WandbLogger(name=f'{hparams.dataset_name}', project='gcg', 
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