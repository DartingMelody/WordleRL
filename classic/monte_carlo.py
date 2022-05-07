from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

import warnings
warnings.simplefilter("error")

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################

    nS = env_spec.nS
    nA = env_spec.nA
    gamma = env_spec.gamma

    Q = np.copy(initQ)
    N = np.zeros((nS, nA))

    for traj in trajs:
        G = 0.0
        W = 1.0
        T = len(traj)
        t = T - 1
        while t >= 0:
            step = traj[t]
            s = step[0]
            a = step[1]
            r = step[2]
            G = gamma * G + r
            N[s][a] += 1
            Q[s][a] += ((W * G - Q[s][a]) / N[s][a])
            # print(s, a, G, Q[s][a])
            W = W * pi.action_prob(s, a) / bpi.action_prob(s, a)
            t -= 1

    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################

    nS = env_spec.nS
    nA = env_spec.nA
    gamma = env_spec.gamma

    Q = np.copy(initQ)
    C = np.zeros((nS, nA))

    for traj in trajs:
        G = 0.0
        W = 1.0
        T = len(traj)
        t = T - 1
        while t >= 0:
            step = traj[t]
            s = step[0]
            a = step[1]
            r = step[2]
            G = gamma * G + r
            C[s][a] += W
            # if C[s][a] == 0:
            #     print('Check', W)
            # try:
            Q[s][a] += (W * (G - Q[s][a]) / C[s][a])
            # except RuntimeWarning:
            #     print (s, a, G, Q[s][a], W, C[s][a])
            W = W * pi.action_prob(s, a) / bpi.action_prob(s, a)
            if W == 0:
                break
            t -= 1

    return Q
