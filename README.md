# [SurfDock is a Surface-Informed Diffusion Generative Model for Reliable and Accurate Protein-ligand Complex Prediction](https://doi.org/10.1038/s41592-024-02516-y)


Implementation of SurfDock, by Duanhua Cao & Mingan Chen.

This repository contains all code, instructions and model weights necessary to **Generative Reliable and Accurate Protein-ligand Complex and Screen Compounds** by SurfDock, eval SurfDock or to retrain a new model.
### 🔔 News
1. **Duanhua have performed several engineering optimizations on SurfDock, enabling it to dock multiple molecules in the same batch. Moreover, our code supports multi-GPU parallel computation. Test on 8*H800(80G), our speed about 0.417s/molecule (40 pose with 20 denoise steps). Enjoy it**

2. **We have added an “example.yml” file in params_example dir to our code repository, detailing the parameters used. This is intended to help users select parameters that are best suited for their tasks, whether they wish to retrain the model, fine-tune our model, or use it directly.**
**More infomations in Section 3**

If you have any question, feel free to open an issue or reach out to us: [caodh@zju.edu.cn](caodh@zju.edu.cn).

![Alt Text](figs/docking.gif)

## Section 1 : Dataset

The files in `data` contain the names for the time-based data split.

If you want to train one of our models with the data then:

1. download it from [zenodo](https://zenodo.org/record/6408497)
2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`

## Section 2 : Setup Environment

(in CodeOcean we have setuped the environment, youcan run the bash script to use SurfDock)
Or you can follow the instructions to setup the environment

```bash
conda create -y -n SurfDock python==3.10
source /opt/conda/bin/activate SurfDock
conda install -y --channel=https://conda.anaconda.org/conda-forge --channel=https://conda.anaconda.org/pytorch --channel=https://conda.anaconda.org/pyg mamba && conda clean -ya
mamba install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -y --channel=https://conda.anaconda.org/conda-forge --channel=https://conda.anaconda.org/pytorch --channel=https://conda.anaconda.org/pyg numpy==1.20 scipy==1.8.1 pandas==2.1.2 &&conda clean -ya
mamba install -y --channel=https://conda.anaconda.org/conda-forge --channel=https://conda.anaconda.org/pytorch --channel=https://conda.anaconda.org/pyg openff-toolkit==0.15.2 openmm==8.1.1 openmmforcefields==0.12.0 pdbfixer==1.9 && conda clean -ya
mamba install -y --channel=https://conda.anaconda.org/conda-forge --channel=https://conda.anaconda.org/pytorch --channel=https://conda.anaconda.org/pyg babel==2.13.1 biopandas==0.4.1 openbabel==3.1.1 plyfile==1.0.1 prody==2.4.0 torch-ema==0.3 torchmetrics==1.2.1 && conda clean -ya
mamba install -y pyg -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -U --no-cache-dir spyrmsd scikit-learn==1.3.2 accelerate==0.15.0 biopython==1.79 e3nn==0.5.1 huggingface-hub==0.17.3 mdanalysis==2.4.0 posebusters==0.2.7 rdkit==2023.3.1 tokenizers==0.13.3 transformers==4.29.2 wandb==0.16.1
pip install pymesh
pip install https://github.com/nuvolos-cloud/PyMesh/releases/download/v0.3.1/pymesh2-0.3.1-cp310-cp310-linux_x86_64.whl
mamba install loguru
pip install dimorphite_dl
pip install prefetch_generator
```
#### Tips.1 **if you have some errors about this file ~/yourpath/.../pymesh/lib/libstdc++.so.6，just raname it as libstdc++.so copy.6 or some names like this, since this file not be used in our env**
### Masif & data processed env dependencies
mamba install mx::reduce
mamba install conda-forge::openbabel

## Section 3 :  We prepared two examples as appetizers for users , users can follow these scripts to use SurfDock as a SBDD Tool.
1. Please setup the env dependencies, then cd ~/bash_scripts/test_scripts
2. Just run the eval_sample.sh script and screen_samples.sh for test SurfDock 

|   Device |   Speed  | Test Samples  | Sampling conformers/molecule | Output conformers/molecule | Sampling steps|
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|  1*H800  |   ~3.260s/molecule  |   100  |   40  |   1  |   20  |
|  8*H800  |   ~0.414s/molecule  |   800  |   40  |   1  |   20  |
|  1*H800  |   ~3.284s/molecule  |   100  |   40  |   40  |   20  |
|  8*H800  |   ~0.417/molecule  |   800  |   40  |   40  |   20  |

## Section 4 : Running SurfDock on your complexes

In our code directory, we write many bash scripts to help user to run SurfDock ,you just follow the example file to adjust the file path ,and then run the specific bash file !
pleas run:
`cd /root/capsule/SurfDock/bash_scripts/test_scripts/`
`bash eval_samples.sh`
if you want to use the wandb to record the result ,please set the next parameters:
    --wandb 
    --wandb_key your_wandb_key 
    --wandb_name your_wandb_name 
Finanly, check the result in the relative path (which you can find in the Docking.sh)

## Section 5: Running SurfDock to Screen compund

a sample run:
`cd /root/capsule/SurfDock/bash_scripts/test_scripts/`
`bash screen_samples.sh`
if you want to use the wandb to record the result ,please set the next parameters:
    --wandb 
    --wandb_key your_wandb_key 
    --wandb_name your_wandb_name 
Finanly, check the result (a csv file include the result DIR) in the out_dir path (which you can find in the score_inplace.sh)

## Section 6 : Retraining SurfDock

If you want to retrain your SurfDock, please Download the data and place it as described in the "Dataset" section above.
since we need esm model to get embedding, we need to install esm model before retarining the SurfDock

`git clone https://github.com/facebookresearch/esm `
`cd esm`
`pip install -e .`

since we also need the surface information about the protein ,so you can folloing the next links to get the surface information

https://github.com/OptiMaL-PSE-Lab/DeepDock
https://github.com/LPDI-EPFL/masif

You can follow the steps in /root/capsule/SurfDock/bash_scripts/test_scripts/eval.sh to get surface information.

Then, you can prepare the esm embedding file for the protein by run the next commands:
`cd /root/capsule/code/SurfDock/bash_scripts/train_SurfDock_docking_module`
Make sure you set up all parameter files in the esm_embedding.sh 
` bash esm_embedding.sh`

then 
`cd /root/capsule/code/SurfDock/bash_scripts/train_SurfDock_docking_module`
Make sure you set up all parameter files in the train_SurfDock.sh
` bash train_SurfDock.sh`
if you want to use the wandb to record the result ,please set the next parameters in train_SurfDock.sh:
    --wandb 
    --wandb_key your_wandb_key 
    --wandb_name your_wandb_name 

## Section 7 : Retraining SurfScore

Same as the previous step


## Section 8 : Citation
Cao, D., Chen, M., Zhang, R. et al. SurfDock is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction. Nat Methods (2024). 
https://doi.org/10.1038/s41592-024-02516-y


Cao D, Chen M, Zhang R, et al. SurfDock is a Surface-Informed Diffusion Generative Model for Reliable and Accurate Protein-ligand Complex Prediction[J]. bioRxiv, 2023: 2023.12. 13.571408.

## Section 9 : License

MIT