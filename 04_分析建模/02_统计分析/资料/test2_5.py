import nsfg
import thinkstats2
import thinkplot
import numpy as np
df=nsfg.ReadFemPreg()
nsfg.CleanFemPreg(df)
live=df[df.outcome==1]
firsts=live[live.birthord==1]
others=live[live.birthord!=1]
first_hist=thinkstats2.Hist(firsts.prglngth)
others_hist=thinkstats2.Hist(others.prglngth)
width=0.45
thinkplot.preplot(2)
thinkplot.Hist(first_hist,align='right',width=width)
thinkplot.Hist(others_hist,align='left',width=width)
#thinkplot.config(axis=[27,46,0,2800])
thinkplot.Show(xlabel='weeks',ylabel='frequency',axis=[21,46,0,2800])