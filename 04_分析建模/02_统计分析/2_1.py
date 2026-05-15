import nsfg
import thinkstats2
import thinkplot
import numpy as np

# preg = nsfg.ReadFemPreg()
# live = preg[preg.outcome == 1]

df=nsfg.ReadFemPreg()
nsfg.CleanFemPreg(df)
birthwgt_lb=df.birthwgt_lb
#生成总体重的真方图
hist=thinkstats2.Hist(birthwgt_lb,label='birthwgt_lb')


# print(thinkplot.Mode(hist))
# print(thinkplot.AllModes(hist))
print("2.一个分布的众数就是他的最频繁值.\n变量birthwgt_lb中最常出现的值为:")
print(thinkstats2.Mode(hist))
print("3.针对变量birthwgt_lb,返回一个值-频数对列表.\n列表按频数降序排列为:")
print(thinkstats2.AllModes(hist))

for val,freq in hist.Items():
    print(val,freq)
thinkplot.Hist(hist)
thinkplot.show(xlabel='pounds',ylabel='frequency')



