#Environments

Our method is a combination of **Codeformer** and **Diffbir**, so the environment configuration entirely follows the setup of **Codeformer** and **Diffbir**. Complete the environment configuration as follows:
```sh
conda create -n NTIRE-team04 python=3.10
conda activate NTIRE-team04
pip install -r /PATH/TO/YOUR/NTIRE2025_RealWorld_Face_Restoration/models/team10_cfDiffbir/DiffBIR/requirements.txt
pip install -r /PATH/TO/YOUR/NTIRE2025_RealWorld_Face_Restoration/models/team10_cfDiffbir/CodeFormer/requirements.txt
```