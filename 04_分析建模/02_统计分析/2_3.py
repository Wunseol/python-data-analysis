# agepreg
import nsfg
import thinkstats2
import thinkplot
import numpy as np

# preg = nsfg.ReadFemPreg()
# live = preg[preg.outcome == 1]

df=nsfg.ReadFemPreg()
nsfg.CleanFemPreg(df)
# live = df[df.outcome == 1]
# print(df)

agepreg=df.agepreg*100

# agepreg.Index.astype(agepreg, int, copy=True)
# agepreg=agepreg.index.astype(agepreg,int)

#生成总体重的真方图
hist=thinkstats2.Hist(agepreg,label='agepreg')

for val,freq in hist.Items():
    print(val,freq)
thinkplot.Hist(hist)
thinkplot.show(xlabel='age',ylabel='frequency')
