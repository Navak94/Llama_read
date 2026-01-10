
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


df = pd.read_csv(r'SchData2015.csv', low_memory = False)

print(df.head(2))
print(df.tail(3))
print(type(df))
print(df.shape)
print(df.info())
print(df.describe())


df = df.fillna(df.mean(numeric_only = True))


num_cols = ['SSSNo',
            'PatientType',
            'SchedulePriority',
            'PatientAge',
            'ScheduledCaseDuration',
            'TotalSurgeryMin',
            'PatientInRoomMin',
            'SetUpMin',
            'CleanUpMin',
            'SchCreateDays',
            'OTS']

df2 = df[num_cols]
print(df2.head(5))


corr = df2.corr()
print(corr)


sn.heatmap(corr)
plt.title('Correlation Heatmap ~ SchData2015 Numeric Variables')
plt.tight_layout()
plt.savefig('SchData2015_corr_heatmap.png')
plt.show()


