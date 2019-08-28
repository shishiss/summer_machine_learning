from matplotlib import pyplot as plt
import numpy as np
import math

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
label_dict = {0: 'Not doanting_blood', 1: 'Donating_blood'}
feature_dict = {i:label for i,label in zip(
                range(4),
                  ('Recency',
                  'Frequency',
                  'Monetary',
                  'Time', ))}

def get_Data(filename):
	charactor = []
	tag = []
	with open(filename) as txtData:
		lines = txtData.readlines()
		for line in lines:
			line = line.strip().split(',')
			charactor.append([float(tk) for tk in line[0:-1]])
			tag.append(int(line[-1]))
		#print(charactor[0:5], tag[0:5])
	return np.array(charactor), np.array(tag)
file_path = './blood_data.txt'
#X,y都是numpy.ndarray型的
X, y = get_Data(file_path)
    
for ax,cnt in zip(axes.ravel(), range(4)):  

    # set bin sizes
    min_b = math.floor(np.min(X[:,cnt]))
    max_b = math.ceil(np.max(X[:,cnt]))
    bins = np.linspace(min_b, max_b, 25)

    # plottling the histograms
    for lab,col in zip(range(0, 2), ('yellow', 'green')):
        ax.hist(X[y==lab, cnt],
                   color=col,
                   label='class %s' %label_dict[lab],
                   bins=bins,
                   alpha=0.5,)
    ylims = ax.get_ylim()

    # plot annotation
    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims)+2])
    ax.set_xlabel(feature_dict[cnt])
    ax.set_title('charactoristic #%s' %str(cnt+1))

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')

fig.tight_layout()       

plt.show()