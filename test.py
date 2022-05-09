
import os
import subprocess
import shlex
import argparse
import sys

def subrun(com, capture_output=False):
    stdout = None if not capture_output else subprocess.PIPE
    universal_newlines = None if not capture_output else True
    return subprocess.run(shlex.split(com), stdout=stdout, universal_newlines=universal_newlines, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', '-w', required=False, type=str)
    parser.add_argument('--dataset', '-dt', choices=['smallset', 'wordspace'], required=False, type=str, default='wordspace')
    parser.add_argument('--policy', '-p', choices=['MC', 'TD0', 'NstepSarsa', 'bandits', 'random', 'greedy'], required=False, type=str, default='MC')
    parser.add_argument('--type', '-t', choices=['Sarsa', 'ExpectedSarsa', 'Qlearning'], required=False, type=str, default='Qlearning')
    args = parser.parse_args()
    epsilon = 0.2
    if args.policy == 'random' or args.policy == 'greedy':
        if args.policy == 'random':
            epsilon = 1
        else:
            epsilon = 0
        args.policy = 'bandits'
    run_command = "python "+args.policy+".py --dataset "+args.dataset
    if args.word:
        run_command = run_command + " --word "+ args.word
    if args.policy == "TD0":
        run_command = run_command + " --policy "+ args.type
    if args.policy == "bandits":
        run_command = run_command + " --epsilon "+ str(epsilon)
    assert(subrun(run_command).returncode == 0)
