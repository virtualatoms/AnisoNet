GitHub[https://github.com/virtualatoms/AnisoNet] | Paper [https://arxiv.org/abs/2405.07915]

AnisoNet is an equivariant graph neural network used to predict the dielectric tensor of crystal materials. 

## Installation
```
git clone https://github.com/virtualatoms/AnisoNet.git
cd AnisoNet
```
Then to install the packages, run:
```
pip install -e .
```
## Content of AnisoNet
You can find all source code in `src/anisonet`, all the code to generate the plots used in the paper in `notebooks/plots` and train anisonet from scratch by running `scripts/run_train.sh`. 

## Citation
If you use AnisoNet in your work, please cite it as follows:
```
@misc{lou_discovery_2024,
	title = {Discovery of highly anisotropic dielectric crystals with equivariant graph neural networks},
	urldate = {2024-05-22},
	publisher = {arXiv},
	author = {Lou, Yuchen and Ganose, Alex M.},
	month = may,
	year = {2024},
}
```

## Acknowledgements
We thank Jason Munro for help with obtaining the dielectric tensor dataset from the Materials Project. A.M.G. was supported by EPSRC Fellowship EP/T033231/1. We are grateful to the UK Materials and Molecular Modelling Hub for computational resources, which are partially funded by EPSRC (EP/T022213/1, EP/W032260/1 and EP/P020194/1). This project made use of time on the Tier 2 HPC facility JADE, funded by EPSRC (EP/P020275/1).