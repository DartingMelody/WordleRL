from re import X
import numpy as np


class StateActionFeatureVectorWithTile():
    def __init__(self, state_low: np.array, state_high: np.array,
                 num_actions: int, num_tilings: int, tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        self.tile_width = tile_width
        self.dimensions = np.size(state_low)

        num_tiles_along_dim = np.ceil(
            np.divide(np.subtract(state_high, state_low), tile_width)) + 1
        print(num_tiles_along_dim)
        self.weight_offset = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            self.weight_offset[i] = np.prod(num_tiles_along_dim[i + 1:])
        self.num_tiles = int(np.prod(num_tiles_along_dim))
        print(self.num_tiles)

        # num_tilings X num_tiles_in_tiling
        self.weights = np.random.rand(
            num_actions * num_tilings * self.num_tiles) - 0.5

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return (self.num_actions * self.num_tilings * self.num_tiles)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        feature_vector = np.zeros(self.feature_vector_len())
        if done:
            return feature_vector

        for tiling_index in range(self.num_tilings):
            state_tile = np.floor(
                np.divide(
                    np.subtract(
                        s,
                        np.subtract(
                            self.state_low, tiling_index / self.num_tilings *
                            self.tile_width)), self.tile_width))
            # print(s, state_tile)
            # print(tiling_index, self.num_tiles,
            #       np.dot(state_tile, self.weight_offset))
            features_per_action = self.num_tilings * self.num_tiles
            feature_vector[features_per_action * a +
                           tiling_index * self.num_tiles +
                           int(np.dot(state_tile, self.weight_offset))] = 1

        return feature_vector


def SarsaLambda(
    env,  # openai gym environment
    gamma: float,  # discount factor
    lam: float,  # decay rate
    alpha: float,  # step size
    X: StateActionFeatureVectorWithTile,
    num_episode: int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    #TODO: implement this function
    for e in range(num_episode):
        s = env.reset()
        done = False
        action = epsilon_greedy_policy(s, done, w)
        x = X(s, done, action)
        z = 0
        q_old = 0
        while True:
            s, reward, done, _ = env.step(action)
            action = epsilon_greedy_policy(s, done, w)
            x_prime = X(s, done, action)
            q = np.dot(w, x)
            q_prime = np.dot(w, x_prime)
            delta = reward + gamma * q_prime - q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += (alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x)
            q_old = q_prime
            x = x_prime

            if done:
                # Episode complete
                if e % 100 == 0:
                    print('Completed episode ' + str(e))
                break

    return w