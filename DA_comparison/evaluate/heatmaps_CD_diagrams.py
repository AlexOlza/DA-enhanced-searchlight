from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
path = '../../figures/CD_diagrams/'
df = pd.read_csv(path+'table_CD_diagrams.csv',index_col=0)
print(df)

fig, ax = plt.subplots()
df_totals = df.copy()

df_totals.iloc[:-1, :-1] = np.nan
df_totals.iloc[-1, -1] = np.nan
sns.heatmap(df.drop('Total',axis=1).drop('Total',axis=0),ax=ax,annot=True, fmt='g')
sns.heatmap(df_totals, cbar=False, linecolor='lightgray', ax=ax,annot = True, fmt = '.0f', cmap = 'Blues')

plt.tight_layout()
plt.savefig(path+'heatmap_CD_diagrams.png')

