import numpy as np

from util import accuracy
from hmm import HMM


# Utility function to get state_dict
def get_state_dict(tags):
	state_dict = {}
	i = 0
	for tag in tags:
		state_dict[tag] = i
		i += 1

	return state_dict


# Utility function to calculate pi values
def get_pi(train_data, tags, state_dict):
	pi = np.zeros(len(tags))

	initial_tags = []
	for line in train_data:
		initial_tags.append(line.tags[0])

	#print ("initial_tags:", initial_tags)
	pi_dict = {}
	for each_tag in tags:
		pi_dict[each_tag] = 0

	for i in initial_tags:
		pi_dict[i] += 1

	#print ("pi_dict:", pi_dict)
	total_initial_states = len(initial_tags)

	for key, value in pi_dict.items():
		index = state_dict[key]
		pi[index] = float(value)/float(total_initial_states)

	return pi


# Utility function to get transition table A
def get_A(train_data, tags, S, state_dict):
	A = np.zeros([S,S], dtype=float)

	for each_line in train_data:
		tags = each_line.tags
		for tag in range(len(tags)-1):
			s_from = state_dict[tags[tag]]
			s_to = state_dict[tags[tag+1]]
			A[s_from][s_to] += 1.0

	A = A/A.sum(axis=1)[:,None]

	return A


# Utility function to get emission table B
def get_obs_dict_B(train_data, tags, S, L, state_dict):
	all_words = []
	all_tags = []

	for line in train_data:
		for index in range(line.length):
			all_words.append(line.words[index])
			all_tags.append(line.tags[index])

	i = 0

	all_uniq_words = set(all_words)
	obs_dict = {}
	for each_word in all_uniq_words:
		obs_dict[each_word] = i
		i += 1
	#print ("obs_dict:", obs_dict)

	B = np.zeros([S, len(all_uniq_words)], dtype=float)

	for index in range(len(all_words)):
		B[state_dict[all_tags[index]]][obs_dict[all_words[index]]] += 1

	B = B / B.sum(axis=1)[:, None]
	#print ("B", B)
	return obs_dict, B

def model_training(train_data, tags):
    """
    Train HMM based on training data
    
    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags
    
    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    S = len(tags)
    L = 0
    for line in train_data:
        L = L + line.length
    
    state_dict = get_state_dict(tags)
    
    pi = get_pi(train_data, tags, state_dict)
    
    A = get_A(train_data, tags, S, state_dict)
    
    obs_dict, B =  get_obs_dict_B(train_data, tags, S, L, state_dict)
    
    model = HMM(pi, A, B, obs_dict, state_dict)
    return model

def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class
    
    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []

    S = len(model.pi)
    LOW_PROB = 10**-6

	# Handle unknown observations that were not seen while training
    for line in test_data:
        for word in line.words:
            if word not in model.obs_dict:
                new_obs_prob = np.full(S, LOW_PROB)
                model.B = np.c_[model.B, new_obs_prob]
                model.obs_dict[word] = len(model.obs_dict)
                
    for line in test_data:
        Osequence = line.words
        tagging.append(model.viterbi(Osequence))
    return tagging
