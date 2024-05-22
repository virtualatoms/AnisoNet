import warnings
warnings.filterwarnings("ignore")
import torch
from lightning import seed_everything
from e3nn.io import CartesianTensor
import pandas as pd
from anisonet.data import BaseDataset
from anisonet.model import E3nnModel
from anisonet.train import BaseLightning
from lightning.pytorch import Trainer
from pytorch_lightning.loggers import CSVLogger


seed_everything(1234)
torch.multiprocessing.set_sharing_strategy('file_system')
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def main() -> None:
    """
    This script runs the training for AnisoNet
    """
    args = tools.parser().parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    """
    This script runs training for AnisoNet
    """       

    ct = CartesianTensor("ij=ji")
    df = pd.read_pickle(args.train_file)
    df = df[df["dielectric_scalar"] < 15]
    df = df.reset_index(drop=True)
    df.rename({"dielectric_irreps": "target"}, axis=1, inplace=True)

    dataset = BaseDataset(df, cutoff=5)

    net = E3nnModel(
        in_dim=118,                     # dimension of one-hot node feature
        in_attr_dim=118,                # dimension of one-hot node attribute
        em_dim=args.em_dim,             # dimension of node feature embedding
        em_attr_dim=args.em_dim,        # dimension of node attribute embedding
        irreps_out=str(ct),             # output irrep shape
        layers=args.layers,             # number of gate layers
        mul=args.mul,                   # multiplicity of features after each layers
        lmax=args.lmax,                 # highest l for spherical harmonics
        max_radius=dataset.cutoff,
        number_of_basis=args.num_basis, # basis for radial embedding
        num_neighbors=dataset.num_neigbours,
        reduce_output=True,
        same_em_layer=True              # whether to use the same embedding layer for one-hot atom type and one-hot atomic mass
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model = BaseLightning(
        dataset,
        net,
        batch_size=args.batch_size,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    trainer = Trainer(
        max_epochs=args.max_epoch,
        accelerator="gpu",
        logger=CSVLogger(".", name=args.name),
        enable_progress_bar=args.enable_progress_bar,
        strategy="ddp_spawn",
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()


