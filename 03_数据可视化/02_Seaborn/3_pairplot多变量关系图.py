# 数据来源: StudentsPerformance.csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

StudentsPerformance = pd.read_csv(Path(__file__).parent / 'StudentsPerformance.csv')

sns.histplot(StudentsPerformance['math score'])
plt.show()
