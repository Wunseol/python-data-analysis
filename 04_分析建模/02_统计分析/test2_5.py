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

def WeightDifference(live, firsts, others):
    """
        第一个婴儿和其他婴儿的体重差异。
        live:所有活产的数据框架
        firsts:第一个婴儿的数据框架
        others:其他的DataFrame
    """
    mean0 = live.totalwgt_lb.mean()
    mean1 = firsts.totalwgt_lb.mean()
    mean2 = others.totalwgt_lb.mean()

    var1 = firsts.totalwgt_lb.var()
    var2 = others.totalwgt_lb.var()

    print('平均值')
    print('First babies 第一个婴儿平均值：', mean1)
    print('Others 其他婴儿平均值：', mean2)

    print('方差')
    print('First babies 第一个婴儿方差：', var1)
    print('Others 其他婴儿方差：', var2)

    print('Difference in lbs （磅差）：', mean1 - mean2)
    print('Difference in oz （盎司差）', (mean1 - mean2) * 16)

    print('Difference relative to mean (%age points) 相对于平均值的差异（%年龄点）', 
          (mean1 - mean2) / mean0 * 100)

    d = thinkstats2.CohenEffectSize(firsts.totalwgt_lb, others.totalwgt_lb)
    print('计算Cohen d的值为:', d)

WeightDifference(live, firsts, others)



#thinkplot.config(axis=[27,46,0,2800])
thinkplot.Show(xlabel='weeks',ylabel='frequency',axis=[21,46,0,2800])