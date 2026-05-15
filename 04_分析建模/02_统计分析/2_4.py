import nsfg
import thinkstats2
import thinkplot
import numpy as np

# preg = nsfg.ReadFemPreg()
# live = preg[preg.outcome == 1]

df=nsfg.ReadFemPreg()
nsfg.CleanFemPreg(df)
totalwgt_lb=df.totalwgt_lb
#生成总体重的真方图
hist=thinkstats2.Hist(totalwgt_lb,label='totalwgt_lb')

for val,freq in hist.Items():
    print(val,freq)
thinkplot.Hist(hist)
thinkplot.show(xlabel='totalwgt',ylabel='frequency')
