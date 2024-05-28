"""Defines the training function for Anisonet."""

import warnings

import click
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from lightning.pytorch import Trainer
from pytorch_lightning.loggers import CSVLogger

from anisonet.data import BaseDataset
from anisonet.model import E3nnModel
from anisonet.train import BaseLightning

warnings.filterwarnings("ignore")

seed_everything = torch.manual_seed(1234)
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


@click.command()
@click.option(
    "--name",
    default="AnisoNet_default",
    help="Name of the folder your model will be saved in",
    required=True,
)
@click.option(
    "--train_file",
    default="dataset/train_dataset.p",
    help="Training file, should be train.p",
    required=True,
)
@click.option(
    "--em_dim",
    default=48,
    help="Embedding dimension of node attribute and node features",
    required=True,
    type=int,
)
@click.option(
    "--layers",
    default=2,
    help="Number of gate layers in the model",
    required=True,
    type=int,
)
@click.option(
    "--lmax",
    default=3,
    help="Maximum order of spherical harmonics in the model",
    required=True,
    type=int,
)
@click.option(
    "--num_basis", default=15, help="Number of radial basis", required=True, type=int
)
@click.option(
    "--mul",
    default=48,
    help="Multiplicity of irreps after each gate layer",
    required=True,
    type=int,
)
@click.option("--lr", default=0.003, help="Learning rate", required=True, type=float)
@click.option("--wd", default=0.03, help="Weight decay", required=True, type=float)
@click.option("--batch_size", default=12, help="Batch size", required=True, type=int)
@click.option(
    "--max_epoch", default=120, help="Number of maximum epochs", required=True, type=int
)
@click.option("--enable_progress_bar", help="Enable progress bar?", type=bool)
@click.option(
    "--distributed",
    default=False,
    help="Enable distributed training",
    type=bool,
    required=False,  # this is optional but makes the intention clear
)
def train(
    name,
    train_file,
    em_dim,
    layers,
    lmax,
    num_basis,
    mul,
    lr,
    wd,
    batch_size,
    max_epoch,
    enable_progress_bar,
    distributed,
):
    """Train anisonet."""
    click.echo("Starting training with the following parameters:")
    click.echo(f"Model Name: {name}")
    click.echo(f"Training File: {train_file}")

    # Data preprocessing and loading
    ct = CartesianTensor("ij=ji")
    train_dataset = pd.read_pickle(train_file)
    dataset = BaseDataset(train_dataset, cutoff=5)

    # Model initialization
    net = E3nnModel(
        in_dim=118,
        in_attr_dim=118,
        em_dim=em_dim,
        em_attr_dim=em_dim,
        irreps_out=str(ct),
        layers=layers,
        mul=mul,
        lmax=lmax,
        max_radius=dataset.cutoff,
        number_of_basis=num_basis,
        num_neighbors=dataset.num_neighbors,
        reduce_output=True,
        same_em_layer=True,
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Lightning module setup
    model = BaseLightning(
        dataset, net, batch_size=batch_size, optimizer=optimizer, scheduler=scheduler
    )

    # Training
    print("Cuda availability is", torch.cuda.is_available())
    if distributed:
        torch.multiprocessing.set_sharing_strategy("file_system")
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="gpu",
            logger=CSVLogger(".", name=name),
            enable_progress_bar=enable_progress_bar,
            strategy="ddp_spawn",
        )
    else:
        trainer = Trainer(
            max_epochs=max_epoch,
            accelerator="gpu",
            logger=CSVLogger(".", name=name),
            enable_progress_bar=enable_progress_bar,
        )
    trainer.fit(model)
