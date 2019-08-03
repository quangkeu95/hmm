import numpy as np
from hidden_markov import hmm

state = ('0', '1')
observation = ('0', '1', '2')

init_prob = np.matrix('0.6 0.4')
transition_prob = np.matrix('0.7 0.3; 0.4 0.6')
observation_prob = np.matrix('0.5 0.4 0.1; 0.1 0.3 0.6')

test = hmm(state, observation, init_prob, transition_prob, observation_prob)

observation_seq = ('1', '1', '2', '0', '0')

print("State sequence")
print(test.viterbi(observation_seq))