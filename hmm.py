from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        alpha[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for i in range(1, L):
            alpha[:, i] = self.B[:, self.obs_dict[Osequence[i]]] * np.sum((self.A.T * alpha[:, i - 1]).T, axis=0)
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        beta[:, L-1] = np.ones(S)
        for i in range(L - 1)[::-1]:
            for j in range(S):
                beta[j][i] = np.sum(self.A[j, :] * self.B[:, self.obs_dict[Osequence[i+1]]] * beta[:, i + 1])
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:, alpha.shape[1]-1])
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)

        prob = (alpha * beta) / seq_prob
        return prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)

        for t in range(L - 1):
            for j in range(S):
                for k in range(S):
                    prob[j][k][t]= [self.A[j , k].T * alpha[j, t] * self.B[k , self.obs_dict[Osequence[t+1]]] * beta[ k, t + 1]]/ seq_prob
        return prob
    
    
    def get_key(self, val):
        for key, value in self.state_dict.items():
            if val == value:
                return key

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        Delta = np.zeros([S, L])
        delta[:,0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]

        for i in range(1, L):
            for j in range(S):
                max_prob = self.A[:,j] * delta[:,i-1]
                delta[j, i] = self.B[j, self.obs_dict[Osequence[i]]] * np.max(max_prob)
                Delta[j, i] = np.argmax(max_prob)


        path.append(np.argmax(delta[:, L - 1]))

        for i in range(1, L)[::-1]:
            index = int(path[0])
            path.insert(0, Delta[index][i])

        for i in range(len(path)):
            path[i] = self.get_key(int(path[i]))
        return path
