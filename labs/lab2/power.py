import numpy as np

def p_value(sample1, sample2, reps, size, t_obs):
    count = 0
    
    sample = np.concatenate((sample1, sample2))
    
    for it in range(reps):
        s = np.random.choice(sample, size=size*2, replace=False)
        s1 = s[:size]
        s2 = s[size:]
        
        t_perm = s2.mean() - s1.mean()
        
        if t_perm > t_obs:
            count += 1
            
    return count / reps

def power(sample1, sample2, reps, size, alpha):
    t_obs = sample2.mean() - sample1.mean()
    count = 0
    
    for it in range(reps):
        s1 = np.random.choice(sample1, size=size, replace=True)
        s2 = np.random.choice(sample2, size=size, replace=True)
        p_val = p_value(s1, s2, reps, size, t_obs)
        
        if p_val < (1 - alpha):
            count += 1
            
    return count / reps