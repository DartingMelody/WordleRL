import gym
import gym_wordle
import numpy as np
from state import WORDS, State, word2action
from gym_wordle.exceptions import InvalidWordException


class ValueFeatureVector():
    def __init__(self, state_dims):
        self.state_len = state_dims
        self.weights = np.random.rand(self.state_len) - 0.5

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.state_len

    def __call__(self, s: State, done) -> np.array:
        """
        implement function x: S+ -> [0,1]^d
        if done is True, then return 0^d
        """
        feature_vector = np.zeros(self.feature_vector_len())
        if done:
            return feature_vector

        return s.state


def TDLambda(
    env,  # openai gym environment
    gamma: float,  # discount factor
    lam: float,  # decay rate
    alpha: float,  # step size
    X: ValueFeatureVector,
    num_episode: int,
) -> np.array:
    """
    True online Sarsa(\lambda) with Value Function Approximation
    """
    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        next_words, next_states = s.possible_actions()
        V = [np.dot(w, X(s_prime, done)) for s_prime in next_states]

        if np.random.rand() < epsilon:
            choice = np.random.randint(len(next_words))
        else:
            choice = np.argmax(V)
        return next_words[choice], next_states[choice]

    w = np.zeros((X.feature_vector_len()))
    success = 0

    for e in range(num_episode):
        _ = env.reset()
        next_word = "stare"
        s = State()
        s.from_word(next_word)
        done = False
        x = X(s, done)
        z = 0
        v_old = 0
        while True:
            print(next_word)
            # Perform action
            obs, reward, done, _ = env.step(word2action(next_word))
            # print(obs)

            # Copy over state information and current observation
            s_prime = State()
            s_prime.copy_state(s)
            s_prime.from_obs(s, next_word, obs)
            s = s_prime

            # Choose next action to perform
            next_word, s_next = epsilon_greedy_policy(s, done, w)

            # Update Logic
            x_prime = X(s, done)
            v = np.dot(w, x)
            v_prime = np.dot(w, x_prime)
            delta = reward + gamma * v_prime - v
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += (alpha * (delta + v - v_old) * z - alpha * (v - v_old) * x)
            v_old = v_prime
            x = x_prime
            s = s_next

            if done:
                # Episode complete
                if reward == 1:
                    success += 1
                if e % 100 == 0 and e != 0:
                    print('Completed episode ' + str(e))
                    print("Successes: " + str(success) + "/" + str(e) + ": " +
                          str(success / e))
                break

    print("Successes: " + str(success) + "/" + str(num_episode) + ": " +
          str(success / num_episode))

    return w


# Train on the entire dataset
env = gym.make('Wordle-v0')
s = State()
TDLambda(env, 1., 0.8, 0.01, ValueFeatureVector(s.state_len), len(WORDS))
