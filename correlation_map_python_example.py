
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


df = pd.read_csv(r'./data_sets/HDI_2025.csv', low_memory = False)

print(df.head(2))
print(df.tail(3))
print(type(df))
print(df.shape)
print(df.info())
print(df.describe())


df.columns = df.columns.str.strip()

num_cols = [
    'HDI rank',
    'Human Development Index (HDI)',
    'Life expectancy at birth',
    'Expected years of schooling',
    'Mean years of schooling',
    'Gross national income (GNI) per capita',
    'GNI per capita rank minus HDI rank'
]

df2 = df[num_cols].copy()

# Convert everything to numeric; invalid stuff like '..' becomes NaN
df2 = df2.apply(pd.to_numeric, errors='coerce')

# Now fill missing numeric values (optional)
df2 = df2.fillna(df2.mean())

corr = df2.corr()


sn.heatmap(corr)
plt.title('HDI Correlation Heatmap')
plt.tight_layout()
plt.savefig('HDI_heatmap.png')
plt.show()


