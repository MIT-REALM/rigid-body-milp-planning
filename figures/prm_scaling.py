import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
# import seaborn as sns

import matplotlib

# matplotlib.rcParams['text.usetex'] = True

# to avoid IEEE Type 3 font error in pdf
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontsize = 32
parameters = {
	'font.family': 'Times New Roman',
	'axes.labelsize': fontsize,
	'axes.titlesize': fontsize,
	'xtick.labelsize': fontsize,
	'ytick.labelsize': fontsize,
	'legend.fontsize': fontsize
}
plt.rcParams.update(parameters)

# length of path
time_results = {
	'clength-1': [11.6, 115.8, 109.8],   # bounds[-2, 14; -3, 9]
	'clength-0.6': [13.4, 92.0, 67.3],
	'2d-STL2': [92.0, 67.3, 89.0],
	'2d-STL3': [70.5, 92.0, 67.3],
}

# node number
std_results = {
	'clength-1': [92.0, 0.1, 0.8],
	'clength-0.6': [0, 92.0, 67.3],
	'2d-STL2': [92.0, 67.3, 89.0],
	'2d-STL3': [70.5, 92.0, 67.3],
}

color = {
	'ours': '#8D9637',
	'PRM-0.3': '#4E5A5A',
	'PRM-0.12': '#DDCEB9',
}

method_list = ['ours', 'PRM-0.3', 'PRM-0.12']
scenario_list = ['clength-1', 'clength-0.6', '2d-STL2', '2d-STL3']

gs = gridspec.GridSpec(1, 1, width_ratios=[1])
fig = plt.figure(figsize=(11, 5))  # (22, 5)

######################## length of path ########################
ax1 = fig.add_subplot(gs[0, 0])
for i in range(len(method_list)):
	ax1.plot(np.arange(len(scenario_list)),
			 [time_results[scene][i] for scene in scenario_list],
			 color=color[method_list[i]], linewidth=3)
	ax1.fill_between(np.arange(len(scenario_list)),
					 [time_results[scene][i] - std_results[scene][i] for scene in scenario_list],
					 [time_results[scene][i] + std_results[scene][i] for scene in scenario_list],
					 color=color[method_list[i]], alpha=0.3)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Online time (ms)')
plt.setp(ax1, xticks=np.arange(len(scenario_list)), xticklabels=scenario_list)
# for label in ax1.get_xticklabels():
# 	label.set_rotation(20)

fig.legend(method_list, loc='upper center', ncol=6, bbox_to_anchor=(0.49, 1.120), columnspacing=1.0)
# plt.savefig('obs_planning.pdf', bbox_inches='tight')
plt.show()
