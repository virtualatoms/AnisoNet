GitHub[https://github.com/virtualatoms/AnisoNet] | Paper [https://arxiv.org/abs/2405.07915]

AnisoNet is an equivariant graph neural network used to predict the dielectric tensor of crystal materials.

## Installation
```
git clone https://github.com/virtualatoms/AnisoNet.git
```
Then to install the packages, run:
```
pip install -e .
```
To train AnisoNet:
```
#!/bin/bash
anisonet-train --name "anisonet" \
               --train_file "../dataset/train_dataset.p" \
               --em_dim 48 \
               --layers 2 \
               --lmax 3 \
               --num_basis 15 \
               --mul 48 \
               --lr 0.003 \
               --wd 0.03 \
               --batch_size 12 \
               --max_epoch 120 \
               --enable_progress_bar True
```

## Content of AnisoNet
You can find all source code in `src/anisonet`, all the code to generate the plots used in the paper in `notebooks/plots` and train anisonet from scratch by running `scripts/run_train.sh`.

## Citation
If you use AnisoNet in your work, please cite it as follows:
```
@misc{lou_discovery_2024,
	title = {Discovery of highly anisotropic dielectric crystals with equivariant graph neural networks},
	publisher = {arXiv},
	author = {Lou, Yuchen and Ganose, Alex M.},
	month = may,
	year = {2024},
}
```

## Acknowledgements
We thank Jason Munro for help with obtaining the dielectric tensor dataset from the Materials Project. A.M.G. was supported by EPSRC Fellowship EP/T033231/1. We are grateful to the UK Materials and Molecular Modelling Hub for computational resources, which are partially funded by EPSRC (EP/T022213/1, EP/W032260/1 and EP/P020194/1). This project made use of time on the Tier 2 HPC facility JADE, funded by EPSRC (EP/P020275/1).
