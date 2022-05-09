import gym
import math
import gym_wordle
from typing import Iterable
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical
from state import WORDS, State, word2action

import warnings

warnings.filterwarnings("error")


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(nn.Linear(state_dim, 32),
                                               nn.ReLU(), nn.Linear(32, 32),
                                               nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.linear_relu_stack(x)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(PolicyNet, self).__init__()
        self.relu_softmax_stack = nn.Sequential(nn.Linear(state_dim, 32),
                                                nn.ReLU(), nn.Linear(32, 32),
                                                nn.ReLU(),
                                                nn.Linear(32, num_actions),
                                                nn.Softmax(dim=0))

    def forward(self, x):
        return self.relu_softmax_stack(x)


class PiApproximationWithNN():
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
        # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
        # a_t = tf.constant([1, 2])
        # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
        # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        self.net = PolicyNet(state_dims, num_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=alpha,
                                          betas=(0.9, 0.999))

    def __call__(self, s) -> int:
        # TODO: implement this method
        m = Categorical(self.eval_state(s).detach())
        a = m.sample().item()
        # print(a)
        return a

    def eval_state(self, s):
        self.net.eval()
        s_float32 = np.float32(s)
        v = self.net(torch.tensor(s_float32))
        # print(s, v)
        return v

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        # Compute the loss
        # if isinstance(G, float):
        #     G = torch.tensor([G])
        # else:
        #     G = torch.tensor(G)
        loss = -gamma_t * delta * torch.log(self.eval_state(s)[a])

        self.net.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        self.net = ValueNet(state_dims)
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=alpha,
                                          betas=(0.9, 0.999))
        self.loss_func = torch.nn.MSELoss()

    def __call__(self, s) -> float:
        # TODO: implement this method
        return self.eval_state(s).detach().numpy()[0]

    def eval_state(self, s):
        self.net.eval()
        s_float32 = np.float32(s)
        v = self.net(torch.tensor(s_float32))
        return v

    def update(self, s, G):
        # TODO: implement this methodmethod
        # Compute the loss
        if isinstance(G, float):
            G = torch.tensor([G])
        else:
            G = torch.tensor(G)
        loss = self.loss_func(self.eval_state(s), G)

        self.net.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
        env,  #open-ai environment
        gamma: float,
        num_episodes: int,
        pi: PiApproximationWithNN,
        V: Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G0 = []
    success = 0

    for e in range(num_episodes):
        # Initialize before start of episode
        _ = env.reset()

        # Representation of initial state
        next_word = "stare"
        s = State()
        s.from_word(next_word)

        a = pi(s.state)
        traj = []
        # Complete the episode
        while True:
            print(next_word)
            # Perform action
            obs, reward, done, _ = env.step(word2action(next_word))
            # print(obs)

            traj.append([s, a, reward])

            # Copy over state information and current observation
            s_prime = State()
            s_prime.copy_state(s)
            s_prime.from_obs(s, next_word, obs)

            s = s_prime
            a = pi(s.state)

            if done:
                # Episode complete
                if reward == 1:
                    success += 1
                if e % 100 == 0 and e != 0:
                    print('Completed episode ' + str(e))
                    print("Successes: " + str(success) + "/" + str(e) + ": " +
                          str(success / e))
                break

        T = len(traj)
        for t in range(T - 1, -1, -1):
            G = 0
            for k in range(t + 1, T + 1):
                G += math.pow(gamma, k - t - 1) * traj[k - 1][2]

            s_t = traj[t][0]
            delta = G - V(s_t.state)
            V.update(s_t.state, G)
            pi.update(s_t.state, traj[t][1], math.pow(gamma, t), delta)


# Train on entire dataset
env = gym.make('Wordle-v0')
s = State()
gamma = 1.
alpha = 3e-4
pi = PiApproximationWithNN(s.state_len, len(WORDS), alpha)
B = VApproximationWithNN(s.state_len, alpha)
REINFORCE(env, gamma, len(WORDS), pi, B)
