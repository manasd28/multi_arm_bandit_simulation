import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy.stats as stats

from environment import *
from agents import *

def print_eln(start, end):
    print('\n'*start)
    print("="*80)
    print('\n'*end)

n_adds = 5
n_rounds = 50000

environment = environment(n_adds)

environment.environment_info()
environment.plot_environment()

print_eln(0, 0)

greedy_agent = greedy(n_adds)
print('Pure Greedy Agent\n')

for i in range(n_rounds):
    add = greedy_agent.get_optimal()
    # print(add, end = " ")
    score = environment.get_score(add)
    greedy_agent.update_score(add, score)

greedy_agent.print_scores()
print(f"\nTotal Score for Pure Greedy Agent: {greedy_agent.get_total_score()}\n")
print_eln(0, 0)

exp = 0.2
exp_greedy_agent = greedy_exploration(n_adds, exp)
print(f"Greedy Agent with Exploration {exp}\n")

for i in range(n_rounds):
    add = exp_greedy_agent.get_optimal()
    # print(add, end = " ")
    score = environment.get_score(add)
    exp_greedy_agent.update_score(add, score)

exp_greedy_agent.print_scores()
print(f"\nTotal Score for Exp Greedy Agent: {exp_greedy_agent.get_total_score()}\n")
print_eln(0, 0)

print("UCB agent\n")

ucb_agent = upper_confidence_bound(n_adds)

for i in range(n_rounds):
    add = ucb_agent.get_optimal()
    score = environment.get_score(add)
    ucb_agent.update_score(add, score)
    
ucb_agent.print_scores()
print(f"\nTotal Score for UCB Agent: {ucb_agent.get_total_score()}\n")
print_eln(0, 0)


