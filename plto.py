import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

DF_long = pd.read_csv("a3_mcts/mcts_simulation100_data.csv")
data = DF_long[(DF_long["Metric"] == "win")]
data.loc[:,'Value'] *= 100
ax = sns.lineplot(
    data=data,
    x="Iterations per Move",  # Former index
    y="Value",  # Values to plot
    hue="Starting Board",  # EB or AB
    style="Strategy",  # mini or random
    errorbar = 'sd',
    markers = True)

# After plotting, get all the line objects
lines = ax.lines

# Loop through and adjust alpha for lines belonging to AB
for line in lines:
    label = line.get_label()  # something like 'EB-minimax', 'AB-random', etc.
    print(label)
    if '4' in label or '6' in label:
        line.set_alpha(0.65)

plt.xscale('log')
plt.ylabel(r'% of Games Won')
plt.xlabel('MCTS Iterations per Move')
# plt.title("Performance Metrics by Group and Mode")
plt.tight_layout()
plt.savefig(f'mcts_simulationFIXED.png', dpi = 500)
plt.show()