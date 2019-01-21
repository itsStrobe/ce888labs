import pandas as pd
import numpy as np

def bootstrap(sample, sample_size, iterations, ci=0.95):
    new_samples = np.empty((iterations, sample_size))
    means = np.empty(iterations)
    
    for it in range(iterations):
        new_samples[it] = np.random.choice(sample, size=sample_size, replace=True)
        means[it] = new_samples[it].mean()
    
    data_mean = means.mean()
    lower = np.percentile(means, ((1 - ci) / 2)*100)
    upper = np.percentile(means, (ci + (1 - ci) / 2)*100)
    return data_mean, lower, upper

if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')
    old = df.as_matrix(columns=['Current fleet']).transpose()[0]
    new = df.as_matrix(columns=['New Fleet'])
    new = new[~np.isnan(new).any(axis=1)].transpose()[0]
    
    old_mean, old_lower, old_upper = bootstrap(old, old.size, 100000)
    new_mean, new_lower, new_upper = bootstrap(new, new.size, 100000)
    
    print('Current Fleet')
    print('Mean:', old_mean)
    print('Lower Bound:', old_lower)
    print('Upper Bound:', old_upper)
    print('--------------------')
    print('New Fleet')
    print('Mean:', new_mean)
    print('Lower Bound:', new_lower)
    print('Upper Bound:', new_upper)
    print('--------------------')