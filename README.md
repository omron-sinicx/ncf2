# Official implementation page of Counterfactual Fairness Filter for Fair-Delay Multi-Robot Navigation

<div align="center">
<img src=assets/result.gif width=550>
</div>

- [paper](https://arxiv.org/abs/2305.11465)
- [project page](https://omron-sinicx.github.io/ncf2/)
- [blog](https://medium.com/sinicx/counterfactual-fairness-filter-for-fair-delay-multi-robot-navigation-aamas2023-e209b54c646d)

## Installation
### **venv**
```console
$ python -m venv .**venv**
$ source .venv/bin/activate
(.venv) $ pip install -e .[dev]
```

### Docker container
```console
$ docker-compose build
$ docker-compose up -d dev
$ docker-compose exec dev bash
```

### Docker container with CUDA enabled
```console
$ docker-compose up -d dev-gpu
$ docker-compose exec dev-gpu bash
```

and update JAX modules in the container...

```console
# pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Way to train agent
We use [Hydra](https://hydra.cc/docs/plugins/nevergrad_sweeper/) for hyperparameter management.

Experiments with different agents and environment with different settings can be run from [train_rl.py](/scripts/train_rl.py).

```console
# python scripts/train_rl.py env.num_agents=5 env.level=10 # train Navi-Only agent in an environment with 5 agents and 10 obstacles.
# python scripts/train_rl.py env.num_agents=10 env.level=20 coop_model=ncf2 # train NCF2 agent in an environment with 5 agents and 10 obstacles.
```

## Experiment
In order to experiment in a specific environment, you first need to train `solitary_policy` to measure the delay.

`solitary_policy` can be trained by training a `Navi-Only` agent in an environment with `num_agents=1`. It can be done by following script.
```console
# python scripts/train_rl.py env.num_agents=1 env.level=10 # train solitary_policy in environment with 10 obstacles.
```
Restoring `solitary_policy` makes it possible to calculate delays in training agents such as `NCF2` and `SOTO`.

For details, please see [train_rl.py](/scripts/train_rl.py)

## Citation
```
@inproceedings{asano2023counterfactual,
author = {Asano, Hikaru and Yonetani, Ryo and Nishimura, Mai and Kozuno, Tadashi},
title = {Counterfactual Fairness Filter for Fair-Delay Multi-Robot Navigation},
year = {2023},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
booktitle = {Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
pages = {887–895},
numpages = {9},
keywords = {cooperation, navigation, multi-agent reinforcement learning},
location = {London, United Kingdom},
series = {AAMAS '23}
}
```
