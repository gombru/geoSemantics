import json
import numpy as np
import matplotlib.pyplot as plt
import operator

dataset = '../../../hd/datasets/YFCC100M/'

model_1 = 'YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55'
model_2 = 'YFCC_NCSL_3rdtraining_epoch_3_ValLoss_0.37'

tags = ['MCLL','NCSL']

input_1 = open(dataset + '/precisions_by_freqs/' + model_1 + '.json')
input_2 = open(dataset + '/precisions_by_freqs/' + model_2 + '.json')

def dict_2_2d_array(dict):
    precisions = []
    frequencies = []
    for k,v in dict.items():
        precisions.append(v['precision'])
        frequencies.append(v['test_appearances'])
    s_p = [x for _, x in sorted(zip(frequencies, precisions), reverse=True)]
    frequencies.sort(reverse=True)
    return s_p, frequencies


s_p_1, s_f_1 = dict_2_2d_array(json.load(input_1))
s_p_2, s_f_2 = dict_2_2d_array(json.load(input_2))

num_points = 40
interval_to_average = int(len(s_p_1) / num_points)

s_p_1_2plot = []
s_p_2_2plot = []

s_f_1_2plot = []
s_f_2_2plot = []


for i in range(0,num_points):

    interval_average = sum(s_p_1[i * interval_to_average : (i+1) * interval_to_average]) / interval_to_average
    s_p_1_2plot.append(interval_average)

    interval_average_freq = sum(s_f_1[i * interval_to_average : (i+1) * interval_to_average]) / interval_to_average
    s_f_1_2plot.append(interval_average_freq)

    interval_average = sum(s_p_2[i * interval_to_average : (i+1) * interval_to_average]) / interval_to_average
    s_p_2_2plot.append(interval_average)

    interval_average_freq = sum(s_f_2[i * interval_to_average : (i+1) * interval_to_average]) / interval_to_average
    s_f_2_2plot.append(interval_average_freq)

x = range(len(s_p_1_2plot))

fig, ax = plt.subplots(1, 1, sharex=True)
plt.plot(x,s_p_1_2plot, color='darkgreen', label=tags[0])
plt.plot(x,s_p_2_2plot, color='darkblue', label=tags[1])

my_xticks = []
for i in range(0,num_points):
	my_xticks.append('')
my_xticks[1] = int(s_f_1_2plot[0]) *50
my_xticks[11] = int(s_f_1_2plot[10])*50
my_xticks[21] = int(s_f_1_2plot[20])*50
my_xticks[31] = int(s_f_1_2plot[30])*50
my_xticks[39] = int(s_f_1_2plot[39])*50
plt.xticks(x, my_xticks)

plt.legend(fontsize=10)
plt.xlabel("Frequency of hashtag")
plt.ylabel("Precision at 10")
# plt.title("Hashtag Frequency")
# plt.yscale('log')
plt.ylim([0,0.6])
plt.xlim(0, num_points-1)
plt.tight_layout()
plt.show()

