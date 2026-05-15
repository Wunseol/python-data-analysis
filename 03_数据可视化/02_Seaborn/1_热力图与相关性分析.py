# 依赖库最低版本要求: seaborn>=0.13
# 数据来源: StudentsPerformance.csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Seaborn 是基于 Python 且非常受欢迎的图形可视化库，在 Matplotlib 的基础上，进行了更高级的封装，使得作图更加方便快捷。
# 即便是没有什么基础的人，也能通过极简的代码，做出具有分析价值而又十分美观的图形。
# Seaborn 可以实现 Python 环境下的绝大部分探索性分析的任务，图形化的表达帮助你对数据进行分析，而且对 Python 的其他库（比如 Numpy/Pandas/Scipy）有很好的支持。

data = pd.read_csv(Path(__file__).parent / 'StudentsPerformance.csv')
data.head()

student = data.drop(['gender','race/ethnicity', 'test preparation course','lunch','parental level of education'], axis=1)

student.head()

corelation = student.corr()

# 图一
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns,annot=True)

# 图二
sns.pairplot(student)

# 图三
sns.displot(student)

# 图四
sns.barplot(student)

# 图五
sns.displot(student['math score'])

# 图六
sns.displot(student['reading score'])

# 图七
sns.displot(student['writing score'])

plt.show()
