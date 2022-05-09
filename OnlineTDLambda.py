import argparse
import math
from typing import Optional
import gym
import gym_wordle
import numpy as np
from state import WORDS, State, load_words, word2action
from gym_wordle.exceptions import InvalidWordException


class ValueFeatureVector():
    def __init__(self, state_dims):
        self.state_len = state_dims

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
    mode: int,  # whether running in train mode or test mode
    user_word: Optional[str] = None,  # run for user provided word
    weights: Optional[np.array] = None) -> np.array:
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

    if weights is not None:
        w = weights
    else:
        w = np.zeros((X.feature_vector_len()))
    success = 0
    chances = []

    for e in range(num_episode):
        # Initialize before start of episode
        _ = env.reset(mode=mode, user_word=user_word)

        # Representation of initial state
        next_word = "stare"
        s = State()
        s.from_word(next_word)

        done = False
        x = X(s, done)
        z = 0
        v_old = 0
        chance = 0
        while True:
            # Perform action
            # print(next_word)
            obs, reward, done, _ = env.step(word2action(next_word))
            chance += 1
            # print(obs)

            # Copy over state information and current observation
            s_prime = State()
            s_prime.copy_state(s)
            s_prime.from_obs(s, next_word, obs)
            s = s_prime

            # Choose next action to perform
            next_word, s_next = epsilon_greedy_policy(s, done, w)

            # Update Logic
            if mode == 0:
                # Under training
                x_prime = X(s, done)
                v = np.dot(w, x)
                v_prime = np.dot(w, x_prime)
                delta = reward + gamma * v_prime - v
                z = gamma * lam * z + (1 -
                                       alpha * gamma * lam * np.dot(z, x)) * x
                w += (alpha * (delta + v - v_old) * z - alpha *
                      (v - v_old) * x)
                v_old = v_prime
                x = x_prime

            s = s_next

            if done:
                # Episode complete
                chances.append(chance)
                if reward == 1:
                    success += 1
                if e % 1000 == 0 and e != 0:
                    print('Completed episode ' + str(e))
                    print("Successes: " + str(success) + "/" + str(e) + ": " +
                          str(success / e))
                break

    print("Successes: " + str(success) + "/" + str(num_episode) + ": " +
          str(success / num_episode))
    print("not predicted " + str(num_episode - success))
    print("percentage not predicted " + str(100 - success * 100 / num_episode))
    print("avg solve chances: " + str(np.average(chances)))

    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', '-w', required=False, type=str)
    parser.add_argument('--dataset',
                        '-dt',
                        choices=['smallset', 'wordspace'],
                        required=False,
                        type=str,
                        default='wordspace')
    parser.add_argument('--epsilon',
                        '-e',
                        required=False,
                        type=float,
                        default='0.2')
    args = parser.parse_args()

    # Train on the entire dataset
    load_words(args.dataset)
    env = gym.make('Wordle-v0')
    env.custom_file(args.dataset)
    s = State()

    # Train over 80% of the words
    weights = TDLambda(env, 1., 0.8, 0.01, ValueFeatureVector(s.state_len),
                       int(0.8 * len(WORDS)), 0)

    if args.word:
        # Test for the user word
        TDLambda(env,
                 1.,
                 0.8,
                 0.01,
                 ValueFeatureVector(s.state_len),
                 1,
                 mode=1,
                 user_word=args.word,
                 weights=weights)
    else:
        # Test over 20% of the words
        TDLambda(env,
                 1.,
                 0.8,
                 0.01,
                 ValueFeatureVector(s.state_len),
                 int(0.2 * len(WORDS)),
                 mode=1,
                 weights=weights)
