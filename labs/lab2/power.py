import numpy as np

def p_value(sample1, sample2, reps, size):
    count = 0
    
    sample = np.concatenate((sample1, sample2))
    
    for it in range(reps):
        s = np.random.choice(sample, sample.size * 2, replace=False)
        s1 = s[:sample.size]
        s2 = s[sample.size:]
        
        t_perm = s2.mean() - s1.mean()
        
        if t_perm > size:
            count += 1
            
    return count / reps

# It was stated in the lecture that the size parameter
# refers to the t_obs value and not the sample size.
def power(sample1, sample2, reps, size, alpha):
    count = 0
    
    for it in range(reps):
        s1 = np.random.choice(sample1, sample1.size, replace=True)
        s2 = np.random.choice(sample2, sample2.size, replace=True)
        p_val = p_value(s1, s2, reps, size)
        
        if p_val < (1 - alpha):
            count += 1
            
    return count / reps