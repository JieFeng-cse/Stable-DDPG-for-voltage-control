# Stability Constrained Reinforcement Learning for Real-Time Voltage Control in Distribution Systems
This repository contains source code necessary to reproduce the results presented in the paper with the same title. It also provides an OpenAI Gym environment for training various Reinforcement Learning algorithms in the IEEE-123 bus and IEEE-13 bus test case.<br />
Authors: Jie Feng, Yuanyuan Shi, Guannan Qu, Steven H. Low, Anima Anandkumar, Adam Wierman<br />
The paper is available here (https://arxiv.org/pdf/2209.07669.pdf).
# hyper-parameters
![plot](./hyperparameters1.png)
![plot](./hyperparameters2.png)
# How to train

>python train_DDPG.py --algorithm safe-ddpg --env_name 13bus --status train<br />
#customize your own algorithm, env_name and status<br />
#env: 13bus,123bus,13bus3p<br />
#algorithm: linear, safe-ddpg,ddpg<br />
#status: train,test<br />
#check points are available<br />


Thank you!
