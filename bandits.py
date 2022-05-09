import fnmatch
import random
import numpy as np
import argparse
import math

def sort_words(poss_words, words_pair_dict):
    words_pair_list = []
    for word in poss_words:
        words_pair_list.append((word, words_pair_dict[word]))
    words_pair_list.sort(key=lambda x: x[1])
    poss_words = []
    for word in words_pair_list:
        poss_words.append(word[0])
    return poss_words

def bandit_policy(letters, letters_not, letters_inc_pos, action, words_lst, letters_dict,letters_rep_not, words_pair_dict, epsilon):
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
    prob = random.uniform(0, 1)
    index = 0
    if prob < epsilon and len(poss_words) > 1:
        index  = random.randint(0, len(poss_words)-1)
    # print("index is "+str(index))
    words_lst = poss_words
    return poss_words[index], words_lst


def get_next_state(letters, letters_not, letters_inc_pos, action, dest_word, letters_dict, letters_rep_not):
    letters = []
    letters_not_temp = []
    letters_dict = {}
    destx = ""
    letters_inc_pos =[]
    letters_inc_pos_dict = {}
    dict_dest = {}
    for (i, l) in enumerate(action):
        if(action[i]==dest_word[i]):
            letters.append((l,i))
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
            if (dict_dest[l] == 0):
                del dict_dest[l]
    for (l,i) in letters_not_temp:
        if l not in letters_dict and l not in letters_inc_pos_dict:
            letters_not.append(l)
        else:
            letters_rep_not.append((l,i))
    letters_not = list( dict.fromkeys(letters_not) )
    letters_inc_pos = list( dict.fromkeys(letters_inc_pos) )
    return (letters, letters_not, letters_inc_pos, letters_dict, letters_rep_not)


def train(returns,retun, Q, state_to_actions, pi, iterations, Lines, dataset, epsilon):
    solver = []
    not_predict = 0
    for dest_word in Lines:
        print("DEST WORD is "+dest_word)
        action = 'stare'
        with open('/Users/tarannumkhan/Desktop/WordleRL/'+dataset+'1.txt', 'r') as f:
            Linesf = f.readlines()
        words_lst = []
        words_pair_dict = {}
        for line in Linesf:
            words_lst.append(line.split()[0])
            # print(line.split()[1])
            words_pair_dict[line.split()[0]] = int(line.split()[1])
        letters = []
        letters_not = []
        letters_inc_pos =[]
        letters_dict = {}
        letters_rep_not = []
        i = 0
        while(len(letters)!=5):
            # print("letters are ", letters)
            # print("letters incorrect pos are ", letters_inc_pos)
            # print("incorrect letters are ", letters_not)
            # print("incorrect letters rep are ", letters_rep_not)
            if(i!=0):
                action, words_lst = bandit_policy(letters, letters_not, letters_inc_pos, action, words_lst, letters_dict, letters_rep_not, words_pair_dict, epsilon)
            print("on chance "+str(i+1)+" ACTION is "+action)
            (letters, letters_not, letters_inc_pos, letters_dict,letters_rep_not) = get_next_state(letters, letters_not, letters_inc_pos, action, dest_word, letters_dict, letters_rep_not)
            if(len(letters) == 5):
                solver.append(i+1)
                break
            if(i==5):
                not_predict = not_predict + 1
            i = i + 1
    print("avg solve chances " + str(sum(solver)/len(solver)))
    print("percentage not predicted "+str((not_predict/len(solver))*100))
    print("not predicted "+str(not_predict))
    return (returns,retun, Q, state_to_actions, pi)

def test(returns,retun, Q, state_to_actions, pi, Lines, dataset, epsilon):
    solver = []
    not_predict = 0
    for dest_word in Lines:
        print("DEST WORD is "+dest_word)
        action = 'stare'
        with open('/Users/tarannumkhan/Desktop/WordleRL/'+dataset+'1.txt', 'r') as f:
            Linesf = f.readlines()
        words_lst = []
        words_pair_dict = {}
        for line in Linesf:
            words_lst.append(line.split()[0])
            # print(line.split()[1])
            words_pair_dict[line.split()[0]] = int(line.split()[1])
        letters = []
        letters_not = []
        letters_inc_pos =[]
        letters_dict = {}
        letters_rep_not = []
        i = 0
        while(len(letters)!=5):
            # print("letters are ", letters)
            # print("letters incorrect pos are ", letters_inc_pos)
            # print("incorrect letters are ", letters_not)
            # print("incorrect letters rep are ", letters_rep_not)
            if(i!=0):
                action, words_lst = bandit_policy(letters, letters_not, letters_inc_pos, action, words_lst, letters_dict, letters_rep_not, words_pair_dict, epsilon)
            print("on chance "+str(i+1)+" ACTION is "+action)
            (letters, letters_not, letters_inc_pos, letters_dict,letters_rep_not) = get_next_state(letters, letters_not, letters_inc_pos, action, dest_word, letters_dict, letters_rep_not)
            if(len(letters) == 5):
                solver.append(i+1)
                break
            if(i==5):
                not_predict = not_predict + 1
            i = i + 1
    print("avg solve chances " + str(sum(solver)/len(solver)))
    print("percentage not predicted "+str((not_predict/len(solver))*100))
    print("not predicted "+str(not_predict))
    return (returns,retun, Q, state_to_actions, pi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', '-w', required=False, type=str)
    parser.add_argument('--dataset', '-dt', choices=['smallset', 'wordspace'], required=False, type=str, default='wordspace')
    parser.add_argument('--epsilon', '-e', required=False, type=float, default='0.2')
    args = parser.parse_args()
    dest_word = args.word
    dataset = args.dataset
    epsilon = args.epsilon
    returns = []
    retun = {}
    Q = {}
    state_to_actions = {}
    pi = {}
    iterations = 2
    play = False
    with open('/Users/tarannumkhan/Desktop/WordleRL/'+dataset +'.txt', 'r') as f:
        Lines = f.readlines()
        random.shuffle(Lines)
    split_n = math.ceil(0.8 * len(Lines))
    (returns,retun, Q, state_to_actions, pi) = train(returns,retun, Q, state_to_actions, pi, iterations, Lines[:int(split_n)], dataset, epsilon)
    if dest_word:
        play = True
    if play == False:
        (returns,retun, Q, state_to_actions, pi) = test(returns,retun, Q, state_to_actions, pi, Lines[int(split_n):], dataset, epsilon)
    else:
        Lines = []
        Lines.append(dest_word)
        print("playing")
        (returns,retun, Q, state_to_actions, pi) = test(returns,retun, Q, state_to_actions, pi, Lines, dataset, epsilon)