import numpy as np

class greedy:
    
    def __init__(self, n_adds):
        self.n_adds = n_adds
        self.scores = np.full((n_adds), np.Inf)
        
    def get_optimal(self):
        return np.argmax(self.scores)+1
    
    def update_score(self, add, score):
        add-=1
        try:
            assert add>=0 and add<self.n_adds
            
            if self.scores[add] == np.Inf:
                self.scores[add] = score
            else:
                self.scores[add] += score
        except Exception:
            print("The add no. exceeds the add limit")
    
    def get_total_score(self):
        return np.sum(self.scores)
    
    def print_scores(self):
        print("The scores are as follows\n")
        for add in range(1, self.n_adds+1):
            print(f"Add {add}: ", self.scores[add-1])
            
class greedy_epsilon:
    
    def __init__(self, n_adds, exp):
        self.exp = exp
        self.n_adds = n_adds
        self.scores = np.full((n_adds), np.Inf)
        
    def get_optimal(self):
        rnd = np.random.random()
        if(rnd<self.exp):
            return np.random.randint(0, self.n_adds)+1
        else:
            return np.argmax(self.scores)+1
    
    def update_score(self, add, score):
        add-=1
        try:
            assert add>=0 and add<self.n_adds
            
            if self.scores[add] == np.Inf:
                self.scores[add] = score
            else:
                self.scores[add] += score
        except Exception:
            print("The add no. exceeds the add limit")
    
    def get_total_score(self):
        return np.sum(self.scores)
    
    def print_scores(self):
        print("The scores are as follows\n")
        for add in range(1, self.n_adds+1):
            print(f"Add {add}: ", self.scores[add-1])
            
class upper_confidence_bound:
    
    def __init__(self, n_adds):
        self.n_adds = n_adds
        self.scores = np.zeros((n_adds))
        self.selections = np.zeros((n_adds))
        self.turn = 0;
        
    def get_optimal(self):
        upper_bounds = ([np.inf if self.selections[i] == 0 else 
                         self.scores[i]/self.selections[i] + 
                         np.sqrt(3/2 * np.log(self.turn + 1) / self.selections[i])
                       for i in range(self.n_adds)])
        
        return np.argmax(upper_bounds)+1
    
    def update_score(self, add, score):
        add-=1 
        try:
            assert add>=0 and add<self.n_adds
            
            if self.scores[add] == np.Inf:
                self.scores[add] = score
            else:
                self.scores[add]+=score
            
            self.turn+=1
            self.selections[add]+=1
        except Exception:
            print("The add no. exceeds the add limit")
    
    def get_total_score(self):
        return np.sum(self.scores)
        
    def print_scores(self):
        print("The scores are as follows\n")
        for add in range(1, self.n_adds+1):
            print(f"Add {add}: ", self.scores[add-1])
            