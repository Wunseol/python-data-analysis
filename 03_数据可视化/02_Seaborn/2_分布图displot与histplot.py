# 数据来源: StudentsPerformance.csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv(Path(__file__).parent / 'StudentsPerformance.csv')
data.head()
print(data)

student = data.drop(['race/ethnicity', 'parental level of education'], axis=1)

student.head()

print(student)

for k in data:
    print(k)

math = data['math score']
reading = data['reading score']
writing = data['writing score']

print('columns :\n', math)
print('values :\n', reading)
print('values :\n', writing)

# [已弃用] distplot 在 seaborn v0.14 中已移除，请使用 histplot 或 displot 替代
sns.histplot(math)

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(hspace=0.5)

plt.show()
