
import time
import argparse
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import wandb
from pytorch_lightning.loggers import WandbLogger

from graph_gen.data import DATASETS
from graph_gen.data.data_utils import train_val_test_split
from graph_gen.data.orderings import order_graphs, ORDER_FUNCS

from data.dataset import K2TreeDataset
from model.autoencoder import Autoencoder


class TrainAE(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainAE, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.set_max_input_size()
        self.setup_model(hparams)
        # self.ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
        # wandb.config['ts'] = self.ts
    
    def set_max_input_size(self):
        shape_list = [tensor.shape[0] for tensor in self.total_dataset.tree_leaf_tensors]
        self.input_size = max(shape_list)
    
    def setup_datasets(self, hparams):
        data_name = hparams.dataset_name
        order = hparams.order
        
        # map dataset and split train / test / validation
        graph_getter, num_rep = DATASETS[data_name]
        if data_name == 'zinc250k':
            graphs = graph_getter(zinc_path='resource/zinc.csv')
        elif data_name == 'peptides':
            graphs = graph_getter(peptide_path='')
        else:
            graphs = graph_getter()

        order_func = ORDER_FUNCS[order]
        
        
        self.train_graphs, self.val_graphs, self.test_graphs = train_val_test_split(graphs)
        
        # map order graphs
        ordered_graphs = []
        for graphs in [self.train_graphs, self.val_graphs, self.test_graphs]:
            ordered_graph = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, seed=hparams.replicate)
            ordered_graphs.append(ordered_graph)
            
        train_graphs_ord, val_graphs_ord, test_graphs_ord = ordered_graphs
        self.total_dataset = K2TreeDataset([*train_graphs_ord, *test_graphs_ord, *val_graphs_ord], data_name)
        self.train_dataset, self.val_dataset, self.test_dataset = [K2TreeDataset(graphs, data_name) for graphs in ordered_graphs]

        
    def setup_model(self, hparams):
        self.model = Autoencoder(hparams.k, self.input_size, hparams.emb_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=K2TreeDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=K2TreeDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=K2TreeDataset.collate_fn,
            num_workers=self.hparams.num_workers
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
        criterion = MSELoss()
        decoded = logits[1].reshape((-1,self.input_size, hparams.k))
        loss = criterion(decoded, batched_data)
        
        decoded_output = torch.where(decoded>0.5, 1, 0)
        acc_matrix = decoded_output == batched_data
        acc = acc_matrix.sum().item() / torch.flatten(acc_matrix).shape[0]
        total_acc = acc_matrix.sum(dim=[1,2]) == (acc_matrix.shape[1]*acc_matrix.shape[2])
        statistics["loss"] = loss
        statistics["acc"] = acc
        statistics["total_acc"] = sum(total_acc) / len(decoded)

        return loss, statistics

    def training_step(self, batched_data):
        loss, statistics = self.shared_step(batched_data)
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            for key, val in statistics.items():
                self.log(f"train/{key}", val, on_step=False, on_epoch=True, logger=True)
                # wandb.log({f"train/{key}": val})

        return loss

    # def validation_step(self, batched_data, batch_idx):
    #     loss, statistics = self.shared_step(batched_data)
    #     if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
    #         for key, val in statistics.items():
    #             self.log(f"val/{key}", val, on_step=False, on_epoch=True, logger=True)
    #         # wandb.log({f"val/{key}": val})
    #     # pass


        
    @staticmethod
    def add_args(parser):

        parser.add_argument("--dataset_name", type=str, default='GDSS_enz')
        parser.add_argument("--group", type=str, default='AE')
        parser.add_argument("--order", type=str, default='C-M')
        parser.add_argument("--replicate", type=int, default=0)
        parser.add_argument("--k", type=int, default=4)
        parser.add_argument("--emb_size", type=int, default=3) 
        parser.add_argument("--max_epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=2)
        parser.add_argument("--wandb_on", type=str, default='disabled')
        
        return parser


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    TrainAE.add_args(parser)

    hparams = parser.parse_args()
    wandb_logger = WandbLogger(name=f'{hparams.dataset_name}', project='gcg', 
                               group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
    wandb.config.update(hparams)

    model = TrainAE(hparams)

    wandb.watch(model)
    
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        logger=wandb_logger
    )
    trainer.fit(model)