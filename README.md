[GitHub](https://github.com/virtualatoms/AnisoNet) | [Paper](https://arxiv.org/abs/2405.07915)

AnisoNet is an equivariant graph neural network used to predict the dielectric tensor of crystal materials.

## Installation

First clone the repository using

```
git clone https://github.com/virtualatoms/AnisoNet.git
cd AnisoNet
```

To install with GPU capability, run
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then to install the packages, run:
```
pip install -e .
```

To train AnisoNet:
```
anisonet-train --name "anisonet" \
               --train_file "dataset/train_dataset.p" \
               --em_dim 48 \
               --layers 2 \
               --lmax 3 \
               --num_basis 15 \
               --mul 48 \
               --lr 0.003 \
               --wd 0.03 \
               --batch_size 12 \
               --max_epoch 120 \
               --enable_progress_bar
```

## Content of AnisoNet
You can find all source code in `src/anisonet`, all the code to generate the plots used in the paper in `notebooks/plots` and train anisonet from scratch by running `scripts/run_train.sh`.

![Figure 6](notebooks/plots/readme.png "Training data vs new anisotropic discoveries")

## Citation
If you use AnisoNet in your work, please cite it as follows:
```
@misc{lou2024discovery,
    title={Discovery of highly anisotropic dielectric crystals with equivariant graph neural networks},
    author={Yuchen Lou and Alex M. Ganose},
    year={2024},
    eprint={2405.07915},
    archivePrefix={arXiv},
    primaryClass={cond-mat.mtrl-sci}
}
```

## Acknowledgements
We thank Jason Munro for help with obtaining the dielectric tensor dataset from the Materials Project. A.M.G. was supported by EPSRC Fellowship EP/T033231/1. We are grateful to the UK Materials and Molecular Modelling Hub for computational resources, which are partially funded by EPSRC (EP/T022213/1, EP/W032260/1 and EP/P020194/1). This project made use of time on the Tier 2 HPC facility JADE, funded by EPSRC (EP/P020275/1).
