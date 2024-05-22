import argparse
import os
from typing import Optional

def parser() -> argparse.ArgumentParser:
    try:
        import configargparse

        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
        )
        parser.add(
            "--config",
            type=str,
            is_config_file=True,
            help="config file to agregate options",
        )
    except ImportError:
        parser = argparse.ArgumentParser()

    
    parser.add_argument("--name", help="name of the folder your model will be saved in", required=True, default="AnisoNet_default")
    parser.add_argument("--train_file", help="training file, should be train.p", required=True, default="dataset/train.p")
    parser.add_argument("--em_dim", help="embedding dimension of node attribute and node features", required=True, default="48")
    parser.add_argument("--layers", help="number of gate layers in the model", required=True, default="2")
    parser.add_argument("--lmax", help="Maximum order of spherical harmonics in the model", required=True, default="3")
    parser.add_argument("--num_basis", help="number of radial basis", required=True, default="15")
    parser.add_argument("--mul", help="multiplicity of irreps after each gate layer", required=True, default="48")
    parser.add_argument("--lr", help="learning rate", required=True, default="0.003")
    parser.add_argument("--wd", help="weight decay", required=True, default="0.03")
    parser.add_argument("--batch_size", help="batch size", required=True, default="12")
    parser.add_argument("--max_epoch", help="number of maximum epochs", required=True, default="120")
    parser.add_argument("--enable_progress_bar", help="enable progress bar?", required=False, default="True")

    return parser