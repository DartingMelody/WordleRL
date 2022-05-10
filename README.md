# WordleRL

Wordle solver using RL. This repository consists of RL algorithms for solving wordle. 

### Prerequisites

To run the agent, the following dependencies are needed:
- PyTorch
- Pip

### Setup

```
git submodule update --init --recursive
cd gym-wordle/ && pip install . && cd ..
```
These commands will set up the gym wordle environment. 

### Usage

For guessing any word through the RL algorithm, the command to run is:
```
python test.py --policy MC --word midst
```
where policy can be selected from [MC, TD0, NstepSarsa, bandits, greedy, OnlineTDLambda].

**Note:**  
1. To run the algorithms on the complete test set, don't specify the word argument.  
2. The default dataset is the larger one, for running the smaller dataset the command is:
```
python test.py --policy MC --dataset smallset --word midst 
```
For running Sarsa, expected Sarsa or Q learning, use the command:
```
python test.py --policy TD0 --word midst --type Sarsa
```
where type can be [Sarsa, ExpectedSarsa, Qlearning].  
As an example, the command to run the training for larger dataset on Monte Carlo algorithm and then run the testing phase, is:
```
python test.py --policy MC
```
