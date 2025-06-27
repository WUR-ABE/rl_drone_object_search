# UAV-based path planning for efficient localization of non-uniformly distributed weeds using prior knowledge: A reinforcement-learning approach
![drone-grid-env](assets/drone-grid-env-with-orthomosaic.gif "drone-grid-env")
> **UAV-based path planning for efficient localization of non-uniformly distributed weeds using prior knowledge: A reinforcement-learning approach**\
> Rick van Essen, Eldert van Henten, and Gert Kootstra
> Paper: https://doi.org/10.1016/j.compag.2025.110651

## About
This is the code belonging to the paper 'UAV-based path planning for efficient localization of non-uniformly distributed weeds using prior knowledge: A reinforcement-learning approach'. It contains the simulation environment and the code to train and reproduce the results of the paper. 

## Installation
Python 3.12 or higher is needed with all dependencies listed in [`requirements.txt`](requirements.txt). Install using:
 
```commandline
pip install --upgrade pip
pip install -e .
export DATA_HOME=<<path to orthomosaic folders>>  # Only needed when running experiment 5
```

Additionally, `Fields2Cover` is needed to calculate the baseline row-by-row flight path. See their [website](https://fields2cover.github.io/index.html) for installation instructions.

Alternatively, you can use the provided [dev container](.devcontainer).

For experiment 5, the orthomosaics can be downloaded from https://doi.org/10.4121/bbe97051-07df-4934-b634-701d91a2075e.

## Usage

### Train RL agent
To train the RL agent, you need to define an experiment containing the simulation configuration and the training parameters. The [`experiments`](experiments/) folder contains the configuration files for the experiments done in the paper. Then use the bash script to train:

```commandline
./run_trainings <<experiment name>>
```

### Evaluate RL agent
To evaluate a trained RL agent, use the [evaluation script](evaluate.py). To reproduce the experiments done in the paper, run:

```commandline
./run_evaluations <<experiment name>>
```

Then use [`create_results_plots.ipynb`](create_results_plots.ipynb) to reproduce the plots.

### Play
To control the drone by keyboard, run:

```commandline
python3 -m drone_grid_env.utils.play --env_args config_file=<<path to config file>>
```

Use ASDW to move the drone 1 step in each direction and L to land the drone (when enabled in the configuration file).

## Citation
If you find this code usefull, please consider citing our paper:
```
@article{vanEssen2025,
    author = {van Essen, Rick and van Henten, Eldert and Kootstra, Gert},
    title = {UAV-based path planning for efficient localization of non-uniformly distributed weeds using prior knowledge: A reinforcement-learning approach},
    journal = {Computers and Electronics in Agriculture}
    volume = {237},
    pages = {110651},
    year = {2025},
    issn = {0168-1699},
    doi = {10.1016/j.compag.2025.110651}
}
```

## Funding
This research is part of the research program SYNERGIA, funding was obtained from the Dutch Research Council (NWO grant 17626), IMEC-One Planet and other private parties.

<img src="assets/wageningen_logo.png" alt="wageningen university logo" height="50"> &nbsp;&nbsp;&nbsp; <img src="assets/synergia_logo_basis.png" alt="synergia logo" height="50">
