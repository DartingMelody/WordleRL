import math
import numpy as np
from policy import Policy

testing_states = np.array([[-.5, 0], [-0.2694817, 0.014904], [-1.2, 0.],
                           [-0.51103601, 0.06101282], [0.48690072,
                                                       0.04923175]])


class ValueFunctionWithApproximation(object):
    def __call__(self, s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self, alpha, G, s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()


def semi_gradient_n_step_td(
    env,  #open-ai environment
    gamma: float,
    pi: Policy,
    n: int,
    alpha: float,
    V: ValueFunctionWithApproximation,
    num_episode: int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    for e in range(num_episode):
        obs = env.reset()
        T = 10000000000000
        t = 0
        R = [0]
        S = [obs]
        while True:
            if t < T:
                action = pi.action(obs)
                obs, reward, done, info = env.step(action)
                # env.render()

                S.append(obs)
                R.append(reward)
                if done:
                    # Episode complete
                    if e % 100 == 0:
                        print('Completed episode ' + str(e))
                        print([V(s) for s in testing_states])
                    T = t + 1

            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += math.pow(gamma, i - tau - 1) * R[i]
                # print('n-step TD return: ', G)
                if tau + n < T:
                    G += math.pow(gamma, n) * V(S[tau + n])
                    # print('non-final n-step TD return: ', G, S[tau + n],
                    #       V(S[tau + n]))
                V.update(alpha, G, S[tau])

            if tau == T - 1:
                break
            t += 1

        # print([V(s) for s in testing_states])
        # input('To run another episode, please press any key')
