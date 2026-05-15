# 数据来源: seaborn 内置数据集 tips
import seaborn as sns
import matplotlib

sns.set_theme()
tips = sns.load_dataset("tips")
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)
matplotlib.pyplot.show()
