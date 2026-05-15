import nsfg
import thinkstats2
import thinkplot
import numpy as np
df=nsfg.ReadFemPreg()
nsfg.CleanFemPreg(df)
totalwgtlb=df.totalwgt_lb
#生成总体重的真方图
hist=thinkstats2.Hist(totalwgtlb,label='birthtotalwgt_lb')
for val,freq in hist.Items():
    print(val,freq)
thinkplot.Hist(hist)
thinkplot.show(xlabel='pounds',ylabel='frequency')