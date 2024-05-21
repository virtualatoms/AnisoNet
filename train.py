import warnings
warnings.filterwarnings("ignore")
import torch
from lightning import seed_everything
from e3nn.io import CartesianTensor
import pandas as pd
from utils_data import load_data
from utils_data import BaseDataset
from utils_model import E3nnModel
from utils_train import BaseLightning
from lightning.pytorch import Trainer
from pytorch_lightning.loggers import CSVLogger


seed_everything(1234)
torch.multiprocessing.set_sharing_strategy('file_system')
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


ct = CartesianTensor("ij=ji")
file_path = 'dataset/train.csv'
df = load_data(file_path,ct)
df = df[df["dielectric_scalar"] < 15]
df = df.reset_index(drop=True)
df.rename({"dielectric_irreps": "target"}, axis=1, inplace=True)
dataset = BaseDataset(df[:200], cutoff=5)


def main():
    net = E3nnModel(
        in_dim=118,                     # dimension of one-hot node feature
        in_attr_dim=118,                # dimension of one-hot node attribute
        em_dim=48,                      # dimension of node feature embedding
        em_attr_dim=48,                 # dimension of node attribute embedding
        irreps_out=str(ct),             # output irrep shape
        layers=2,                       # number of gate layers
        mul=48,                         # multiplicity of features after each layers
        lmax=3,                         # highest l for spherical harmonics
        max_radius=dataset.cutoff,
        number_of_basis=15,             # basis for radial embedding
        num_neighbors=dataset.num_neigbours,
        reduce_output=True,
        same_em_layer=True              # whether to use the same embedding layer for one-hot atom type and one-hot atomic mass
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model = BaseLightning(
        dataset,
        net,
        batch_size=12,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    trainer = Trainer(
        max_epochs=120,
        accelerator="gpu",
        logger=CSVLogger(".", name="stock_models"),
        enable_progress_bar=True,
        strategy="ddp_spawn",
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()


