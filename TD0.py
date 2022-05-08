import fnmatch
import random
import numpy as np
import argparse

epsilon = 0.2
gamma = 0.9
alpha = 0.9
def sort_words(poss_words, words_pair_dict):
    words_pair_list = []
    for word in poss_words:
        words_pair_list.append((word, words_pair_dict[word]))
    words_pair_list.sort(key=lambda x: x[1])
    poss_words = []
    for word in words_pair_list:
        poss_words.append(word[0])
    return poss_words

def wordlist(letters, letters_not, letters_inc_pos, action, words_lst, letters_dict,letters_rep_not, words_pair_dict):
    action_w = "?????"
    for l in letters:
        action_w = action_w[:l[1]] + l[0] + action_w[l[1]+1:]
    filter_lines = fnmatch.filter(words_lst, action_w)
    # print("matching state_w filter_lines are {}", len(filter_lines))
    filter_list = []
    for word in filter_lines:
        should_add = True
        for l in letters_not:
            if l in word:
                should_add = False
                break
        if should_add:
            filter_list.append(word)
    # print("after removing letters_not {}", len(filter_list))
    filter_lines = []
    for word in filter_list:
        should_add = True
        for (l,pos) in letters_inc_pos:
            if word[pos] == l:
                should_add = False
                break
        for (l,pos) in letters_rep_not:
            if word[pos] == l:
                should_add = False
                break
        if should_add:
            filter_lines.append(word)
    # print("after removing letters_inc_pos {}", len(filter_lines))
    poss_actions = []
    poss_actions_dict = {}
    for (l,pos) in letters_inc_pos:
        for i, ch in enumerate(action_w):
            if(ch == '?' and i!=pos):
                action_w1 = action_w
                # state_w1[i] = l
                action_w1 = action_w1[:i] + l + action_w1[i+1:]
                if l not in poss_actions_dict:
                    poss_actions_dict[l] = []
                poss_actions_dict[l].append(action_w1)
                poss_actions.append(action_w1)
    # print("poss states are {}", poss_states)
    poss_words = []
    for l in poss_actions_dict:
        # print("l is "+l)
        poss_action_list = poss_actions_dict[l]
        poss_words_l = []
        for action_str in poss_action_list:
            # print("State_str is "+state_str)
            templist = fnmatch.filter(filter_lines, action_str)
            poss_words_l = poss_words_l + templist
        poss_words_l = list( dict.fromkeys(poss_words_l) )
        # print("temp poss words are {}", len(poss_words_l))
        if(len(poss_words)> 0 ):
            poss_words = list(set(poss_words)&set(poss_words_l))
        else:
            poss_words = poss_words_l
    if (len(poss_words) == 0):
        poss_words = filter_lines
    # print("final poss words are {}", len(poss_words))
    # print(poss_words)
    poss_words = sort_words(poss_words, words_pair_dict) #poss_words are sorted on their q value in desc where the negative of the q_value of word is stored in words_pair_dict 
    # prob = random.uniform(0, 1)
    # index = 0
    # if prob < epsilon and len(poss_words) > 1:
    #     index  = random.randint(0, len(poss_words)-1)
    # print("index is "+str(index))
    words_lst = poss_words
    return words_lst

def get_action(words_lst, pi, state, Q, state_to_actions):
    max_Q = 0
    pi[state] = words_lst[0]
    if state in state_to_actions:
        for action in state_to_actions[state]:
            if(Q[(state, action)] > max_Q):
                max_Q = Q[(state, action)]
                pi[state] = action
            elif(Q[(state, action)] == max_Q):
                max_Q = Q[(state, action)]
                index = 0
                if (len(state_to_actions[state]) > 1):
                    index  = random.randint(0, len(state_to_actions[state])-1)
                pi[state] = state_to_actions[state][index]
    prob = random.uniform(0, 1)
    if prob < epsilon and len(words_lst) > 1:
        index  = random.randint(0, len(words_lst)-1)
        pi[state] = words_lst[index]
    return pi[state]

def get_greedy_action(words_lst, pi, state, Q, state_to_actions):
    if(state not in state_to_actions):
        index = 0
        prob = random.uniform(0, 1)
        if prob < epsilon and len(words_lst) > 1:
            index  = random.randint(0, len(words_lst)-1)
            return words_lst[index]
    if state in state_to_actions:
        max_Q = 0
        for action in state_to_actions[state]:
            if(Q[(state, action)] > max_Q):
                max_Q = Q[(state, action)]
                greedy_action = action
            elif(Q[(state, action)] == max_Q):
                max_Q = Q[(state, action)]
                index = 0
                if len(state_to_actions[state]) > 1:
                    index  = random.randint(0, len(state_to_actions[state])-1)
                greedy_action = state_to_actions[state][index]
        return greedy_action

def get_next_state(letters, letters_not, letters_inc_pos, action, dest_word, letters_dict, letters_rep_not, rewardc):
    letters = []
    letters_not_temp = []
    letters_dict = {}
    destx = ""
    letters_inc_pos =[]
    letters_inc_pos_dict = {}
    dict_dest = {}
    old_reward = rewardc
    reward = 0
    rewardc = 0
    for (i, l) in enumerate(action):
        if(action[i]==dest_word[i]):
            letters.append((l,i))
            reward = reward + 5 #correct letter reward
            rewardc = rewardc + 5
            if l not in letters_dict:
                letters_dict[l] = []
            letters_dict[l].append(i)
        else:
            letters_not_temp.append((l,i))
            # destx = destx + dest_word[i]
            if dest_word[i] not in dict_dest:
                dict_dest[dest_word[i]] = 1
            else:
                dict_dest[dest_word[i]] = dict_dest[dest_word[i]] + 1               
    for (l,i) in letters_not_temp:
        if l in dict_dest:
            letters_inc_pos.append((l,i))
            letters_inc_pos_dict[l] = 1
            dict_dest[l] = dict_dest[l] - 1
            reward = reward + 1 #incorrect letter reward
            if (dict_dest[l] == 0):
                del dict_dest[l]
    for (l,i) in letters_not_temp:
        if l not in letters_dict and l not in letters_inc_pos_dict:
            letters_not.append(l)
        else:
            letters_rep_not.append((l,i))
    letters_not = list( dict.fromkeys(letters_not) )
    letters_inc_pos = list( dict.fromkeys(letters_inc_pos) )
    reward = reward - old_reward
    return (letters, letters_not, letters_inc_pos, letters_dict, letters_rep_not, reward, rewardc)

def get_next_state1(letters, letters_not, letters_inc_pos, action, dest_word, letters_dict, letters_rep_not):
    letters = []
    letters_not_temp = []
    letters_dict = {}
    destx = ""
    letters_inc_pos =[]
    letters_inc_pos_dict = {}
    dict_dest = {}
    reward = 0
    for (i, l) in enumerate(action):
        if(action[i]==dest_word[i]):
            letters.append((l,i))
            reward = reward + 5 #correct letter reward
            if l not in letters_dict:
                letters_dict[l] = []
            letters_dict[l].append(i)
        else:
            reward = reward - 1
            letters_not_temp.append((l,i))
            # destx = destx + dest_word[i]
            if dest_word[i] not in dict_dest:
                dict_dest[dest_word[i]] = 1
            else:
                dict_dest[dest_word[i]] = dict_dest[dest_word[i]] + 1               
    for (l,i) in letters_not_temp:
        if l in dict_dest:
            letters_inc_pos.append((l,i))
            letters_inc_pos_dict[l] = 1
            dict_dest[l] = dict_dest[l] - 1
            reward = reward + 1 #incorrect letter reward
            if (dict_dest[l] == 0):
                del dict_dest[l]
    for (l,i) in letters_not_temp:
        if l not in letters_dict and l not in letters_inc_pos_dict:
            letters_not.append(l)
        else:
            letters_rep_not.append((l,i))
    letters_not = list( dict.fromkeys(letters_not) )
    letters_inc_pos = list( dict.fromkeys(letters_inc_pos) )
    return (letters, letters_not, letters_inc_pos, letters_dict, letters_rep_not, reward)

def state_concat(letters, letters_not, letters_inc_pos, letters_rep_not):
    state = 'L'
    for (l,i) in letters:
        state = state + l + str(i)
    state = state + 'P'
    for (l,i) in letters_inc_pos:
        state = state + l + str(i)
    state =  state + 'I'
    for l in letters_not:
        state = state + l
    state =  state + 'R'
    for (l,i) in letters_rep_not:
        state = state + l + str(i)
    return state


parser = argparse.ArgumentParser()
parser.add_argument('--word', '-w', required=False, type=str)
parser.add_argument('--policy', '-p', choices=['Sarsa', 'Qlearning', 'ExpectedSarsa'], required=True, type=str)
args = parser.parse_args()
dest_word = args.word
policy = args.policy
test = False

if dest_word:
    test = True

returns = []
retun = {}
Q = {}
state_to_actions = {}
pi = {}
if test == True:
    iterations = 1
else:
    iterations = 2
for it in range(iterations):
    solver = []
    with open('/Users/tarannumkhan/Desktop/WordleRL/smallset.txt', 'r') as f:
        Lines = f.readlines()
    if test == True:
        Lines = []
        Lines.append(dest_word)
    not_predict = 0
    for dest_word in Lines: #in each episode
        print("DEST WORD is "+dest_word)
        action = 'stare'
        with open('/Users/tarannumkhan/Desktop/WordleRL/smallset1.txt', 'r') as f:
            Linesf = f.readlines()
        words_lst = []
        words_pair_dict = {}
        for line in Linesf:
            words_lst.append(line.split()[0])
            words_pair_dict[line.split()[0]] = int(line.split()[1])
        letters = []
        letters_not = []
        letters_inc_pos =[]
        letters_dict = {}
        letters_rep_not = []
        reward = 0
        rewardc = 0
        states = []
        actions = []
        rewards = []
        i = 0
        while(len(letters)!=5):
            states.append(state_concat(letters, letters_not, letters_inc_pos, letters_rep_not))
            if(i!=0):
                action = get_action(words_lst, pi, states[i], Q, state_to_actions)
            print("on chance "+str(i+1)+" ACTION is "+action)
            (letters, letters_not, letters_inc_pos, letters_dict,letters_rep_not, reward, rewardc) = get_next_state(letters, letters_not, letters_inc_pos, action, dest_word, letters_dict, letters_rep_not, rewardc)
            # (letters, letters_not, letters_inc_pos, letters_dict,letters_rep_not, reward) = get_next_state(letters, letters_not, letters_inc_pos, action, dest_word, letters_dict, letters_rep_not)
            words_lst = wordlist(letters, letters_not, letters_inc_pos, action, words_lst, letters_dict, letters_rep_not, words_pair_dict)
            actions.append(action)
            # print("letters are ", letters)
            # print("letters incorrect pos are ", letters_inc_pos)
            # print("incorrect letters are ", letters_not)
            # print("incorrect letters rep are ", letters_rep_not)
            # print("state i is "+states[i])
            # print("action is "+action)
            rewards.append(reward)
            if(states[i] not in state_to_actions):
                state_to_actions[states[i]] = []
                state_to_actions[states[i]].append(action)
            elif action not in state_to_actions[states[i]]:
                state_to_actions[states[i]].append(action)
            next_state = state_concat(letters, letters_not, letters_inc_pos, letters_rep_not)
            print("next state is ", next_state)
            if policy == "Sarsa":
                next_action = get_action(words_lst, pi, next_state, Q, state_to_actions)
            elif "Qlearning":
                next_action = get_greedy_action(words_lst, pi, next_state, Q, state_to_actions)
            if((states[i], actions[i]) not in Q):
                Q[(states[i], actions[i])] = 0
            if((next_state, next_action) not in Q):
                Q[(next_state, next_action)] = 0
            if policy == "ExpectedSarsa":
                Q_sum = 0
                n = 0 
                Q_avg = 0
                if next_state in state_to_actions:
                    for action in state_to_actions[next_state]:
                        Q_sum = Q_sum + Q[(next_state, action)]
                        n = n + 1
                    Q_avg = Q_sum/n
                Q[(states[i], actions[i])] = Q[(states[i], actions[i])] + alpha*(reward + gamma*Q_avg - Q[(states[i], actions[i])])
            else:     #for Qlearning and Sarsa
                Q[(states[i], actions[i])] = Q[(states[i], actions[i])] + alpha*(reward + gamma*Q[(next_state, next_action)] - Q[(states[i], actions[i])])
            if(len(letters) == 5):
                solver.append(i+1)
                rewards[-1] = 25
                Q[(states[i], actions[i])] = 0
                break
            if(i==5):
                not_predict = not_predict + 1
                # Q[(states[i], actions[i])] = 0
            i = i + 1
    print("not predicted "+str(not_predict))
    print("avg solve chances " + str(sum(solver)/len(solver)))