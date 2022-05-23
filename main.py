import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from environment import *
from agents import *

def print_eln(start, end):
    print('\n'*start)
    print("="*80)
    print('\n'*end)

n_adds = 5
n_rounds = 10000

environment = environment(n_adds)

environment.environment_info()
environment.plot_environment()

print_eln(0, 0)

def iterate_rounds(agent, verbose = 0, ax = None):
    print(f"{type(agent).__name__} agent:\n")
    scores = []
    for r in range(n_rounds):
        
        add = agent.get_optimal()
        score = environment.get_score(add)

        if r%100 == 0:
            scores.append(score)
            
        agent.update_score(add, score)
    
    if verbose >= 1:
        agent.print_scores()
    
    if verbose == 2:
        linestyles = {'greedy' : "--", 'greedy_epsilon' : ':', 
                     'upper_confidence_bound' : '-'}
        ax.plot(scores, label = type(agent).__name__, linewidth = 1.5,
                linestyle = linestyles[type(agent).__name__])
    
    print(f"\nTotal score of the agent is: {agent.get_total_score()}")
    print_eln(0, 1)
    return agent.get_total_score()

# Score per iteration
plt.figure(dpi = 120, figsize = (12, 6))
ax = plt.gca()
ax.grid(visible=True, which = 'major', axis = 'both', 
             color = '#bbbdbf', linewidth = 1, linestyle = '--')

s1 = iterate_rounds(greedy(n_adds), 2, ax)
s2 = iterate_rounds(greedy_epsilon(n_adds, 0.2), 2, ax)
s3 = iterate_rounds(upper_confidence_bound(n_adds), 2, ax)

ax.legend()
ax.set_title("Scores per 100 iterations")
ax.set_xlabel("Rounds / 100")
ax.set_ylabel("Score")

# Total Score
plt.figure(dpi = 120, figsize = (12, 6))
ax = plt.gca()
ax.grid(visible=True, which = 'major', axis = 'both', 
              color = '#bbbdbf', linewidth = 1, linestyle = '--')

sns.barplot(x = ['Greedy', 'Greedy Epsilon', 'UCB'], 
            y = np.array([s1, s2, s3])-0.99*min([s1, s2, s3]), ax = ax)
ax.set_title('Comparision between top 1% of total scores of the agents', size = 15)
ax.set_xlabel('Agents', labelpad = 15, size = 13)
ax.set_ylabel('Top 1% scores', labelpad = 15, size = 13)

plt.show()