# Permanent-Dropout: Simple Neuron Pruning for accelerating Deep Reinforcement Learning

## Installation
We recommend using Miniconda to create your virtual env.
```
conda create -n dropout_env python=3.11
conda activate dropout_env
git clone https://github.com/hail-mary/permanent-dropout.git
cd permanent-dropout
pip install gymnasium[mujoco] stable-baselines3 pyyaml
```

## Training
```
python main.py
# specify the log directory by adding `--logdir` flag.
python main.py --logdir logs
```

## Evaluation
```
python main.py --eval [PATH_TO_CHECKPOINTS]
# optional: recording, requires pip install "gymnasium[other]"
python main.py --eval [PATH_TO_CHECKPOINTS] --record
```

## Plot results
```
python main.py --plot .\logs\history.json
```

## References
* Averaging Weights Leads to Wider Optima and Better Generalization
* Improving Stability in Deep Reinforcement Learning with Weight Averaging
