"""Classes for loading in a dataset."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset
from tqdm.auto import tqdm

specie_am = [Atom(z).mass for z in range(1, 119)]
am_onehot = torch.diag(torch.tensor(specie_am)).tolist()


@dataclass
class Data:
    """
    Class to contain graph attributes.

    N and M are the number of nodes and edges in the graph, respectively.

    Parameters
    ----------
    node_input : Tensor
        The node features as a (N, n_node_feats) Tensor.
    edge_index : Tensor
        The edge src and dst as a (2, M) Tensor.
    edge_vec : LongTensor
        The edge vectors as a (M, 3) Tensor.
    target : Tensor
        The target property to learn.
    """

    node_input: torch.Tensor
    node_attr: torch.Tensor
    edge_index: torch.LongTensor
    edge_vec: torch.Tensor
    target: torch.Tensor
    idx: int

    def to(self, device, non_blocking=False):
        """Put data on the compute device."""
        for k, v in self.__dict__.items():
            if k == "idx":
                continue
            self.__dict__[k] = v.to(device=device, non_blocking=non_blocking)


@dataclass
class Batch:
    """
    Class to contain batched graph attributes.

    N and M are the number of nodes and edges across all batched graphs,
    respectively.

    G is the number of graphs in the batch.

    Parameters
    ----------
    node_input : Tensor
        The node features as a (N, n_node_feats) Tensor.
    edge_index : Tensor
        The edge src and dst as a (2, M) Tensor.
    edge_vec : LongTensor
        The edge vectors as a (M, 3) Tensor.
    target : Tensor
        The target property to learn, as a (G, 1) Tensor.
    batch : LongTensor
        The graph to which each node belongs, as a (N, ) Tensor.
    """

    node_input: torch.Tensor
    node_attr: torch.Tensor
    edge_index: torch.LongTensor
    edge_vec: torch.Tensor
    target: torch.Tensor
    batch: torch.LongTensor
    idx: torch.LongTensor

    def to(self, device, non_blocking=False):
        """Put batch on the compute device."""
        for k, v in self.__dict__.items():
            self.__dict__[k] = v.to(device=device, non_blocking=non_blocking)


def collate_fn(dataset):
    """
    Collate a list of Data objects and return a Batch.

    Parameters
    ----------
    dataset : MaterialsDataset
        The dataset to batch.

    Returns
    -------
    Batch
        A batched dataset.
    """
    batch = Batch([], [], [], [], [], [], [])
    base_idx = 0
    for i, data in enumerate(dataset):
        batch.node_input.append(data.node_input)
        batch.node_attr.append(data.node_attr)
        batch.edge_index.append(data.edge_index + base_idx)
        batch.edge_vec.append(data.edge_vec)
        batch.target.append(data.target)
        batch.idx.append(data.idx)
        batch.batch.extend([i] * len(data.node_input))
        base_idx += len(data.node_input)
    return Batch(
        node_input=torch.cat(batch.node_input),
        node_attr=torch.cat(batch.node_attr),
        edge_index=torch.cat(batch.edge_index, dim=-1),
        edge_vec=torch.cat(batch.edge_vec),
        batch=torch.LongTensor(batch.batch),
        target=torch.cat(batch.target),
        idx=torch.LongTensor(batch.idx),
    )


class BaseDataset(Dataset):
    """Dataset of materials properties.

    Parameters
    ----------
    filename : str or DataFrame
        The path to the dataset or a pandas dataframe. If supplying a pandas Dataframe
        then the dataset is expected to contain two columns: "structure" containing
        ASE `Atoms` objects or pymatgen `Structure` objects and "target" containing the
        target to predict. If passing a filename, the file is expected to be in json
        format, containing a list of dictionaries, each with the keys "positions"
        (cartesian atomic positions), "cell" (cell lattice parameters), "numbers"
        (atomic numbers of the atoms).
    cutoff : float
        The cutoff radius for searching for neighbors.
    """

    def __init__(
        self, filename: str | Path, cutoff=5, symprec=0.01, graph_type="cutoff"
    ):
        self.cutoff = cutoff
        self.data = []

        num_nodes = 0
        num_neighbors = 0

        self.structures = []
        self.targets = []

        if isinstance(filename, (Path, str)):
            if Path(filename).suffix == ".gz":
                with gzip.open(filename) as f:
                    jsondata = json.load(f)
            else:
                with open(filename) as f:
                    jsondata = json.load(f)

            for entry in jsondata:
                self.structures.append(entry_to_atoms(entry))
                self.targets.append(entry["target"])
        else:
            self.structures = filename["structure"].values.tolist()
            if not isinstance(self.structures[0], Atoms):
                # presume we have pymatgen structure objects and try and convert
                self.structures = [a.to_ase_atoms() for a in self.structures]
            self.targets = filename["target"].values.tolist()

        for idx, (atoms, target) in tqdm(
            enumerate(zip(self.structures, self.targets)), total=len(self.structures)
        ):
            data = atoms_to_data(
                atoms,
                self.cutoff,
                target=target,
                idx=idx,
            )
            num_nodes += len(data.node_input)
            num_neighbors += len(data.edge_vec)
            self.data.append(data)

        self.num_neighbors = num_neighbors / num_nodes
        self.num_nodes = num_nodes / len(self.data)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        return self.data[idx]


def entry_to_atoms(entry):
    """Convert a dataset entry to atoms object."""
    return Atoms(
        positions=entry["positions"],
        cell=entry["cell"],
        numbers=entry["numbers"],
        pbc=[True, True, True],
    )


def atoms_to_data(
    atoms,
    cutoff,
    target=None,
    idx: int = 0,
    graph_type: str = "cutoff",
):
    """Convert an atoms object to a data object.

    Parameters
    ----------
    atoms : Atoms
        An Ase atoms object.
    cutoff : float
        The cutoff radius for neighbor finding.
    target : list or numpy array
        The target value for the structure.
    idx : int
        Index of the sample.
    type : str
        kwarg to specify how to construct the data. (cutoff / voronoi)

    Returns
    -------
    Data
        A custom Data object.
    """
    if graph_type == "cutoff":
        # construct edge_src, edge_dst, edge_vec using max cut-off.
        try:
            import numpy as np
            from pymatgen.optimization.neighbors import find_points_in_spheres

            positions = atoms.get_positions()
            cell = np.array(atoms.get_cell())
            edge_src, edge_dst, images, _ = find_points_in_spheres(
                positions,
                positions,
                r=cutoff,
                pbc=np.array([1, 1, 1], dtype=int),
                lattice=cell,
                tol=1.0e-8,
            )
            edge_vec = positions[edge_dst] - positions[edge_src] + images @ cell
        except ImportError:
            # fall back on slower ase algorithm if pymatgen not installed
            edge_src, edge_dst, edge_vec = neighbor_list(
                "ijD", atoms, cutoff=cutoff, self_interaction=True
            )

    if target:
        if isinstance(target, (float, int)):
            target = [target]
        target = torch.Tensor(target).unsqueeze(0)
    node_input = torch.Tensor([type_onehot(i) for i in atoms.numbers])
    node_attr = torch.Tensor([mass_onehot(i) for i in atoms.numbers])

    return Data(
        node_input=node_input,
        node_attr=node_attr,
        edge_index=torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        ),
        edge_vec=torch.Tensor(edge_vec.tolist()),
        target=target,
        idx=idx,
    )


def type_onehot(number: int, max_number: int = 118):
    """Onehot encode an atom number into the type encoding."""
    embedding = [0.0] * max_number
    embedding[number - 1] = 1.0
    return embedding


def mass_onehot(number: int):
    """One hot encode an atom number into the mass encoding."""
    return am_onehot[number - 1]
