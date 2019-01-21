import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np


def boostrap(sample, sample_size, iterations, ci=0.95):
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
    df = pd.read_csv('./salaries.csv')

    data = df.values.T[1]
    boots = []
    for i in range(100, 100000, 1000):
        boot = boostrap(data, data.shape[0], i)
        boots.append([i, boot[0], "mean"])
        boots.append([i, boot[1], "lower"])
        boots.append([i, boot[2], "upper"])

    df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
    sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")
    
    sns_plot.axes[0, 0].set_ylim(0,)
    sns_plot.axes[0, 0].set_xlim(0, 100000)

    sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
    sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')


    #print ("Mean: %f")%(np.mean(data))
    #print ("Var: %f")%(np.var(data))
	


	