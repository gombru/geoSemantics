import json
import matplotlib.pyplot as plt
import numpy as np
import operator

tags = json.load(open('../../../datasets/YFCC100M/anns/tags_count.json'))

tags_sorted = sorted(tags.items(), key=operator.itemgetter(1))

tags_values = tags.values()
tags_values.sort(reverse=True)

x = range(len(tags_values))

fig, ax = plt.subplots(1, 1, sharex=True)
ax.fill_between(x, 0, tags_values, color='green')
plt.plot(x,tags_values, color='darkgreen')

plt.xlabel("hashtags ordered by frequency")
plt.ylabel("appearances")
plt.title("Hashtag Frequency")
plt.yscale('log')
plt.ylim([0,100000])
plt.xlim([0,100000])
plt.tight_layout()
plt.show()


