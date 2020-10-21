# DRIML: Deep Reinforcement and InfoMax Learning
Code for Deep Reinforcement and InfoMax Learning (Neurips 2020)

**Note**: The repo is under construction right now, things will get added progressively to it as code is optimized/cleaned. For now, the parallelized Procgen code is released for `rlpyt` version of Feb.19 2020, but the goal is to make it compatible for the latest stable version of `rlpyt`.

## Overview of algorithm
<center>
<img src="https://github.com/bmazoure/DRIML/raw/main/DRIML_thumbnail-01.png" alt="Architecture" width="500"/>
</center>

## Prerequisites

- `rlpyt` (commit `a0f1c3045eac1b12d6305b35200139f9ee2a63cd`). Newer commits might throw errors. Goal: rewrite code in latest stable `rlpyt` version.
- `torch`. Latest stable release seems to work.

## Instructions

- Clone the repo
- Run `python main_procgen.py  --lambda_LL "0" --lambda_GL "0" --lambda_LG "0" --lambda_GG "1" --experiment-name "test" --env-name "procgen-bigfish-v0.500" \
                --n_step-return "7" --nce-batch-size "256" --horizon "10000" --algo "c51" --n-cpus "8" --n-gpus "1" --weight-save-interval "-1" --n_step-nce "-2" \
                --frame_stack "3" --nce_loss "InfoNCE_action_loss" --log-interval-steps=1000 --mode "serial"`, for example. Trains DRIML-randk on 500 Bigfish levels.

To cite:
```
@inproceedings{mazoure2020deep,
  title={Deep Reinforcement and InfoMax Learning},
  author={Mazoure, Bogdan and Combes, Remi Tachet des and Doan, Thang and Bachman, Philip and Hjelm, R Devon},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```