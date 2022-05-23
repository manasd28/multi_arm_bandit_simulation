import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class skewnorm:
    def __init__(self):
        self.a = np.random.randint(10, 20)
        self.loc = np.random.randint(20, 25)
    
    def sample(self):
        return int(stats.skewnorm.rvs(self.a, self.loc))
    
    def plot(self, ax, samples):
        data = stats.skewnorm.rvs(self.a, self.loc, size = samples)
        sns.kdeplot(data, ax = ax, fill = True)

class gaussian:
    def __init__(self):
        self.loc = np.random.randint(20, 23)
        self.scale = np.random.random()*1.5
        
    def sample(self):
        return int(np.random.normal(self.loc, self.scale))
    
    def plot(self, ax, samples):
        data = np.random.normal(self.loc, self.scale, size = samples)
        sns.kdeplot(data, ax = ax, fill = True)

class environment:
    
    def __init__(self, n_adds):
        
        self.n_adds = n_adds
        self.adds = []
        
        agent_types = [skewnorm, gaussian]
        
        try:
            assert n_adds>0 and n_adds<=10
            for add in range(n_adds):
                select = np.random.randint(0, 2)
                if select:
                    agent = skewnorm()
                else:
                    agent = gaussian()
                self.adds.append(agent)
        
        except Exception:
            print("The No. of adds should be in the range [0-9]")
    
    def environment_info(self):
        
        print("\nThe environment info. is as follows:\n")
        print(f"The total no. of adds = {self.n_adds}\n")
        for ind, add in enumerate(self.adds):
            print(f"Add {ind+1} ({type(add).__name__} dist): ", end = "")
            if(type(add).__name__ == 'skewnorm'):
                print(f"a: {add.a}, loc: {add.loc}")
            else:
                print(f"loc: {add.loc}, scale: {add.scale}")
        print('\n')
        
    def plot_environment(self):
        
        plt.figure(dpi = 120, figsize=(12, 6))
        ax = plt.gca()
        ax.grid(visible=True, which = 'major', axis = 'both', 
                     color = '#bbbdbf', linewidth = 1, linestyle = '--')
        
        samples = 10000
        
        for add in range(self.n_adds):
            self.adds[add].plot(ax, samples)
        
        labels = ['Add '+str(ind+1) for ind in range(self.n_adds)]
        ax.legend(labels)
        plt.title('Kernel Density Estimate of Adds in Environment')
        
        plt.tight_layout()
        plt.show()
    
    def get_score(self, add):
        add -= 1
        try:
            assert add>=0 and add<=self.n_adds
            return self.adds[add].sample()
        
        except Exception:
            print("The add no. exceeds the add limit")
        
if __name__ == '__main__':
    test_env = environment(5)
    test_env.environment_info()
    test_env.plot_environment()