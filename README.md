# WordleRL

Wordle solver using RL

This repository consists of RL algorithms for solving wordle. 

Steps to run: 
git submodule update --init --recursive
cd gym-wordle/ && pip install . && cd ..
These commands will set up the gym wordle environment. 
For guessing any word through the RL algorithm, the command to run is:
python test.py --policy MC --word midst (try any 5 letter word)
where policy can be selected from [MC, TD0, NstepSarsa, bandits, greedy, OnlineTDLambda] 
For running Sarsa, expected Sarsa or Q learning, use the command:
python test.py --policy MC --word midst --type Sarsa
where type can be [Sarsa, ExpectedSarsa, Qlearning]
The default dataset is the larger one, for running the smaller dataset the command is:
python test.py --policy MC --dataset smallset --word midst 
