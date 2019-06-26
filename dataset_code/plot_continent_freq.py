import json
import matplotlib.pyplot as plt
import operator

continents = json.load(open('../../../datasets/YFCC100M/anns/continent_count.json'))

continents_sorted = sorted(continents.items(), key=operator.itemgetter(1), reverse=True)

continents_values = continents.values()
continents_values.sort(reverse=True)
print(continents_values)
my_xticks = []
x = range(len(continents_values))
for cont in continents_sorted:
    my_xticks.append(cont[0])
plt.xticks(x, my_xticks, rotation=0, size=11)
width = 1/1.5
plt.bar(x, continents_values, width, color="green", align="center")
plt.tight_layout()
plt.title("Number of Images per Continent")
# plt.yscale('log')
plt.ylim([0,10000000])
plt.tight_layout()
plt.show()


