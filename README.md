# DQN FOR POGEMA

## Contents 

This repository contains DQN algorithm for [pogema](https://github.com/AIRI-Institute/pogema). Algorithm uses logger for training on previous experiments and two NNs: target net and policy net. Policy net is being training every training step and once in `TARGET_UPDATE` steps logs into target net for stable learning. File `vis.py` contains script for visualizing results into `.svg` file.
