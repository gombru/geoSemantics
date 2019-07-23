import json
import numpy as np
import matplotlib.pyplot as plt
import operator

dataset = '../../../hd/datasets/YFCC100M/'

model_1 = 'YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55'
model_2 = 'YFCC_NCSL_3rdtraining_epoch_3_ValLoss_0.37'

tags = ['MCLL','NCSL']

input_1 = dataset + '/precisions_by_freqs/' + model_1 + '.json'
input_2 = dataset + '/precisions_by_freqs/' + model_2 + '.json'

def dict_2_2d_array(dict):
    precisions = []
    frequencies = []
    for k,v in dict:
        precisions.append(v['precision'])
        frequencies.append(v['test_appearances'])
    s_p = [x for _, x in sorted(zip(frequencies, precisions), reverse=True)]
    s_f = frequencies.sort(reverse=True)
    return s_p, s_f


s_p_1, s_f_1 = dict_2_2d_array(json.load(input_1))
s_p_2, s_f_2 = dict_2_2d_array(json.load(input_2))


fig, ax = plt.subplots(1, 1, sharex=True)

plt.plot(s_f_1,s_p_1, color='darkgreen')
plt.plot(s_f_2,s_p_2, color='darkblue')

plt.xlabel("Hashtag test set appearances")
plt.ylabel("Precision at 10")
# plt.title("Hashtag Frequency")
# plt.yscale('log')
# plt.ylim([0,1])
# plt.xlim([0,100000])
plt.tight_layout()
plt.show()


