import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(threshold, max_value, threshold_count):

    my_data = pd.read_csv('652_summary_KLUCB.csv').values
    print('shape of dataset is {}'.format(my_data.shape))
    n,_ = my_data.shape
    instance_freq = np.zeros(n)
    pulls = np.zeros(n)
    for i in range(n):
        row = my_data[i]
        instance_freq[i] = (row[2] + row[3])/row[5]
        pulls[i] = row[5]

    instance = instance_freq

    instance = instance / max(instance)


    for i in range(len(instance)):
        if instance[i] >= threshold:
            instance[i] = max_value


    fig = plt.figure()
    plt.hist(instance, bins=15)
    plt.xlabel('expected reward')
    plt.show()
    plt.close(fig)


    count = 0
    for i in range(len(instance)):
        if instance[i] >= threshold_count:
            count += 1

    print('number of arms with mean greater than {} = {}'.format(threshold_count, count))

    np.save('652_contest.npy', instance)

    print('mean of processed instance = {}'.format(np.mean(instance)))


if __name__ == '__main__':

    generate_data(0.8, 1, 0.8)
