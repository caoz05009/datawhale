导入常用的工具包
```python
import pandas as  pd
import numpy as np
import warnings 
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import datetime 
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

```
读取数据
```python
# 设置数据集路径(注意此处为相对路径)
dataset_path = 'Dataset/'
# 读取数据
data_balance = pd.read_csv(dataset_path+'user_balance_table.csv')
```
pandas中 to_datetime函数转换时间格式，dt函数分别年月日等取出来
```python
data_balance['date'] = pd.to_datetime(data_balance['report_date'], format= "%Y%m%d")
data_balance['day'] = data_balance['date'].dt.day
data_balance['month'] = data_balance['date'].dt.month
data_balance['year'] = data_balance['date'].dt.year
data_balance['week'] = data_balance['date'].dt.week
data_balance['weekday'] = data_balance['date'].dt.weekday
```
datatime模块
datetime.datetime	表示日期时间,datetime.timedelta	表示两个date、time、datetime实例之间的时间间隔，分辨率（最小单位）可达到微秒
```python
# 生成测试集区段数据
start = datetime.datetime(2014,9,1)
testdata = []
while start != datetime.datetime(2014,10,1):
    temp = [start, np.nan, np.nan]
    testdata.append(temp)
    start += datetime.timedelta(days = 1)
testdata = pd.DataFrame(testdata)
testdata.columns = total_balance.columns
```
plt.legend(loc)表示图例的位置
```python
# 画出每日总购买与赎回量的时间序列图

fig = plt.figure(figsize=(20,6))
plt.plot(total_balance['date'], total_balance['total_purchase_amt'],label='purchase')
plt.plot(total_balance['date'], total_balance['total_redeem_amt'],label='redeem')

plt.legend(loc='best')
plt.title("The lineplot of total amount of Purchase and Redeem from July.13 to Sep.14")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.show()
```

```python
# 画出4月份以后的时间序列图
total_balance['date'] = pd.to_datetime(total_balance['date'], format="%Y%m%d").dt.date
total_balance_1 = total_balance[total_balance['date'] >= datetime.date(2014,4,1)]
fig = plt.figure(figsize=(20,6))
plt.plot(total_balance_1['date'], total_balance_1['total_purchase_amt'])
plt.plot(total_balance_1['date'], total_balance_1['total_redeem_amt'])
plt.legend()
plt.title("The lineplot of total amount of Purchase and Redeem from April.14 to Sep.14")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.show()
```
```python
# 分别画出每个月中每天购买赎回量的时间序列图

fig = plt.figure(figsize=(15,15))

plt.subplot(4,1,1)
plt.title("The time series of total amount of Purchase and Redeem for August, July, June, May respectively")

total_balance_2 = total_balance[total_balance['date'] >= datetime.date(2014,8,1)]
plt.plot(total_balance_2['date'], total_balance_2['total_purchase_amt'])
plt.plot(total_balance_2['date'], total_balance_2['total_redeem_amt'])
plt.legend()


total_balance_3 = total_balance[(total_balance['date'] >= datetime.date(2014,7,1)) & (total_balance['date'] < datetime.date(2014,8,1))]
plt.subplot(4,1,2)
plt.plot(total_balance_3['date'], total_balance_3['total_purchase_amt'])
plt.plot(total_balance_3['date'], total_balance_3['total_redeem_amt'])
plt.legend()


total_balance_4 = total_balance[(total_balance['date'] >= datetime.date(2014,6,1)) & (total_balance['date'] < datetime.date(2014,7,1))]
plt.subplot(4,1,3)
plt.plot(total_balance_4['date'], total_balance_4['total_purchase_amt'])
plt.plot(total_balance_4['date'], total_balance_4['total_redeem_amt'])
plt.legend()


total_balance_5 = total_balance[(total_balance['date'] >= datetime.date(2014,5,1)) & (total_balance['date'] < datetime.date(2014,6,1))]
plt.subplot(4,1,4)
plt.plot(total_balance_5['date'], total_balance_5['total_purchase_amt'])
plt.plot(total_balance_5['date'], total_balance_5['total_redeem_amt'])
plt.legend()

plt.xlabel("Time")
plt.ylabel("Amount")
plt.show()
```
Seaborn是基于matplotlib的Python可视化库。violinplot与boxplot扮演类似的角色，它显示了定量数据在一个（或多个）分类变量的多个层次上的分布，这些分布可以进行比较。不像箱形图中所有绘图组件都对应于实际数据点，小提琴绘图以基础分布的核密度估计为特征。


```python
# 画出每个翌日的数据分布于整体数据的分布图

a = plt.figure(figsize=(10,10))
scatter_para = {'marker':'.', 's':3, 'alpha':0.3}
line_kws = {'color':'k'}
plt.subplot(2,2,1)
plt.title('The distrubution of total purchase')
sns.violinplot(x='weekday', y='total_purchase_amt', data = total_balance_1, scatter_kws=scatter_para, line_kws=line_kws)
plt.subplot(2,2,2)
plt.title('The distrubution of total purchase')
sns.distplot(total_balance_1['total_purchase_amt'].dropna())
plt.subplot(2,2,3)
plt.title('The distrubution of total redeem')
sns.violinplot(x='weekday', y='total_redeem_amt', data = total_balance_1, scatter_kws=scatter_para, line_kws=line_kws)
plt.subplot(2,2,4)
plt.title('The distrubution of total redeem')
sns.distplot(total_balance_1['total_redeem_amt'].dropna())
plt.show()
```
