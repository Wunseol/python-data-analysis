import nsfg
import thinkstats2
import thinkplot
import numpy as np

# preg = nsfg.ReadFemPreg()
# live = preg[preg.outcome == 1]

df=nsfg.ReadFemPreg()
nsfg.CleanFemPreg(df)
birthwgt_oz=df.birthwgt_oz
#生成总体重的真方图
hist=thinkstats2.Hist(birthwgt_oz,label='birthwgt_oz')
for val,freq in hist.Items():
    print(val,freq)
thinkplot.Hist(hist)
thinkplot.show(xlabel='ounce(oz)',ylabel='frequency')