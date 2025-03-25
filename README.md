# Conformal Rigidity and Topological Obstructions: A Spectral Perspective on Graph Genus

Exploring the relationship between conformal rigidity, spectral properties, and genus of graphs.

## Setup

Choose one of Pyenv or Conda to manage your Python environment.

### With Pyenv

1. Install Python 3.10.12
    - [Install pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)
    - Run `pyenv install 3.10.12`
    - Pyenv should automatically use this version in this directory. If not, run `pyenv local 3.10.12`
2. Create a virtual environment
    - Run `python -m venv ./venv` to create it
    - Run `. venv/bin/activate` when you want to activate it
        - Run `deactivate` when you want to deactivate it
    - Pro-tip: select the virtual environment in your IDE, e.g. in VSCode, click the Python version in the bottom left corner and select the virtual environment
3. Install dependencies with `pip install -r requirements.txt` 

### With Conda

1. Install miniconda or anaconda
    - [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
    - Or [install anaconda](https://docs.anaconda.com/anaconda/install/)
2. Create a virtual environment
    - Run `conda create --prefix ./venv python=3.10.12` to create it
    - Run `conda activate ./venv` when you want to activate it
        - Run `conda deactivate` when you want to deactivate it
    - Pro-tip: select the virtual environment in your IDE, e.g. in VSCode, click the Python version in the bottom left corner and select the virtual environment
3. Install dependencies with `pip install -r requirements.txt` 

## Usage

Open any of the `.ipynb` notebooks in VS Code or Jupyter Notebook to run the code.

- `conformal_rigidity.ipynb` contains the code for checking whether a graph is conformally rigid.
- `spectral_bound.ipynb` contains delta_g plots

## References

- [Conformally Rigid Graphs](https://arxiv.org/abs/2402.11758) by Steinerberger and Thomas
- [Conformal upper bounds for the eigenvalues of the Laplacian and Steklov problem](https://www.sciencedirect.com/science/article/pii/S0022123611002928) by Asma Hassannezhad
