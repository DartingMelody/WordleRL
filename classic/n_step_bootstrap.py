from math import gamma
import math
from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy


def on_policy_n_step_td(env_spec: EnvSpec,
                        trajs: Iterable[Iterable[Tuple[int, int, int,
                                                       int]]], n: int,
                        alpha: float, initV: np.array) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    nS = env_spec.nS
    nA = env_spec.nA
    gamma = env_spec.gamma

    V = np.copy(initV)

    for traj in trajs:
        T = len(traj)
        t = 0
        while True:
            tau = t - n + 1
            if tau >= 0:
                G = np.sum([
                    math.pow(gamma, i - tau - 1) * traj[i - 1][2]
                    for i in range(tau + 1,
                                   min(tau + n, T) + 1)
                ])
                if tau + n < T:
                    G += math.pow(gamma, n) * V[traj[tau + n][0]]
                V[traj[tau][0]] += alpha * (G - V[traj[tau][0]])

            if tau == T - 1:
                break
            t += 1

    return V


def off_policy_n_step_sarsa(env_spec: EnvSpec,
                            trajs: Iterable[Iterable[Tuple[int, int, int,
                                                           int]]], bpi: Policy,
                            n: int, alpha: float,
                            initQ: np.array) -> Tuple[np.array, Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    nS = env_spec.nS
    nA = env_spec.nA
    gamma = env_spec.gamma

    Q = np.copy(initQ)

    for traj in trajs:
        T = len(traj)
        for t in range(T + n - 1):
            tau = t - n + 1
            if tau >= 0:
                # Calculate rho
                rho = 1
                for i in range(tau + 1, min(tau + n, T - 1) + 1):
                    if traj[i][1] == np.argmax(Q[traj[i][0]]):
                        rho *= 1 / bpi.action_prob(traj[i][0], traj[i][1])
                    else:
                        rho = 0
                        break

                G = np.sum([
                    math.pow(gamma, i - tau - 1) * traj[i - 1][2]
                    for i in range(tau + 1,
                                   min(tau + n, T) + 1)
                ])
                if tau + n < T:
                    G += math.pow(gamma,
                                  n) * Q[traj[tau + n][0]][traj[tau + n][1]]
                Q[traj[tau][0]][traj[tau][1]] += alpha * rho * (
                    G - Q[traj[tau][0]][traj[tau][1]])

    class NStepPolicy(Policy):
        def __init__(self, max_action_dict):
            self._mat = max_action_dict

        def action_prob(self, state, action):
            if self._mat[state] == action:
                return 1
            return 0

        def action(self, state):
            return self._mat[state]

        def to_string(self):
            return self._mat

    nstep_policy = {}
    for s in range(nS):
        nstep_policy[s] = np.argmax(Q[s])
    print(nstep_policy)
    pi = NStepPolicy(nstep_policy)

    return Q, pi
