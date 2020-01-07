# 数据分析

## 比赛背景
* 本比赛为房产租金预测问题，根据给定的数据集，预测房屋租金，为一回归问题
* 数据集中的数据类别包括租赁房源、小区、二手房、配套、新房、土地、人口、客户、真实租金等

## 评分函数
* 评价指标为R-Square
* 比较简单的来看，分母其实为一定值，当分子$\hat{y}$与$y_i$越接近时，整体的值就越接近于1，当预测值于真实值完全相等时，结果就是1。

## 对比赛数据做EDA

### 缺失值分析
* 方法一
``` python
  data_train.info()
```
可通过如上代码显示数据集的相关信息。
可知总共有41440行，51列
![20200106182755.png](https://raw.githubusercontent.com/zz7236/image/master/vscode/20200106182755.png)

如上截图可知，pv与uv均有为空的数据。
* 方法二
``` python
 def missing_values(df):
    alldata_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    alldata_na['existNum'] = len(df) - alldata_na['missingNum']
    alldata_na['sum'] = len(df)
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(df)*100
    alldata_na['dtype'] = df.dtypes
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na
```
通过如上函数，可得出相关缺失数据的信息和缺失率，截图如下：
![20200106183301.png](https://raw.githubusercontent.com/zz7236/image/master/vscode/20200106183301.png)

### 特征值分析
1. 单调特征列分析
* 单调特征列在我看来就是这一列是单增或单减的比例是多少。
``` python
def incresing(vals):
    cnt_1 = 0
    cnt_2 = 0
    len_ = len(vals)
    for i in range(len_-1):
        if vals[i+1] > vals[i]:
            cnt_1 += 1
        if vals[i+1] < vals[i]:
            cnt_2 += 1
    return max(cnt_1,cnt_2)

for col in data_train.columns:
    cnt = incresing(data_train[col].values)
    if cnt / data_train.shape[0] >= 0.55:
        print('单调特征：',col)
        print('单调特征值个数：', cnt)
        print('单调特征值比例：', cnt / data_train.shape[0])
```
![20200106190226.png](https://raw.githubusercontent.com/zz7236/image/master/vscode/20200106190226.png)

* 注：由于ID特征本身本来就应该是一个单调特征，所以不算在内。
2. 特征nunique分布
代码如下：
``` python
for feature in data_train.columns:
    print(feature + "的特征分布如下：")
    print(data_train[feature].value_counts())
    if feature != 'communityName': 
        plt.title(feature)
        plt.hist(data_train[feature], bins=3)
        plt.show()
```
即可知
* rentType 有四种特征值，绝大多数为未知方式
* houseType 中绝大多少都在3室以下
* houseFloor 分为高、中、低三种，分布比较平均
* totalFloor 中绝大多少都在20层以下，10层左右最多
* houseToward 中绝大多数在南北朝向
* houseDecoration 分为四种，其他、精装两类最多
* buildYear 绝大多数分布在2000年左右
* saleSecHouseNum 绝大多少为0
* subwayStationNum 主要是0-7区间内较多
* interSchoolNum 主要集中在0-4区间
3. 统计特征值出现频次大于100的特征
代码如下所示：
```
feature_100 = {
    'name':[],
    'counts':[]
}
for feature in data_train.columns:
    df_value_counts = pd.DataFrame(data_train[feature].value_counts())
    df_value_counts = df_value_counts.reset_index()
    df_value_counts.columns = [feature, 'counts']
    feature_100['name'].append(feature)
    feature_100['counts'].append(df_value_counts[df_value_counts['counts'] >= 100].shape[0])
print(pd.DataFrame(feature_100))
```
所得结果为：
![20200106222814.png](https://raw.githubusercontent.com/zz7236/image/master/vscode/20200106222814.png)

### Label分布
代码如下:
```
fig,axes = plt.subplots(2,3,figsize=(20,5))
fig.set_size_inches(20,12)
sns.distplot(data_train[(data_train['tradeMoney']>0)&(data_train['tradeMoney']<=100000)]['tradeMoney'],ax=axes[0][0])
sns.distplot(data_train[(data_train['tradeMoney']<=20000)]['tradeMoney'],ax=axes[0][1])
sns.distplot(data_train[(data_train['tradeMoney']>20000)&(data_train['tradeMoney']<=50000)]['tradeMoney'],ax=axes[0][2])
sns.distplot(data_train[(data_train['tradeMoney']>50000)&(data_train['tradeMoney']<=100000)]['tradeMoney'],ax=axes[1][0])
sns.distplot(data_train[(data_train['tradeMoney']>100000)]['tradeMoney'],ax=axes[1][1])
plt.show()
```
![20200106224239.png](https://raw.githubusercontent.com/zz7236/image/master/vscode/20200106224239.png)
由本图可知：
绝大多数的租金集中在0-25000之间，
* 在0-20000区间，峰值出现在5000，
* 在20000-50000区间，峰值出现在25000，
* 在50000-125000区间，峰值出现在55000

### 不同的特征值的样本的label的分布

``` python 
fig,axes = plt.subplots(2,3,figsize=(20,5))
fig.set_size_inches(20,12)
temp = data_train[data_train['houseFloor'] == '高']
temp1 = data_train[data_train['houseFloor'] == '中']
temp2 = data_train[data_train['houseFloor'] == '低']
sns.distplot(temp['tradeMoney'],ax=axes[0][0])
sns.distplot(temp1['tradeMoney'],ax=axes[0][1])
sns.distplot(temp2['tradeMoney'],ax=axes[0][2])
plt.show()
```
![20200106231030.png](https://raw.githubusercontent.com/zz7236/image/master/vscode/20200106231030.png)

当houseFloor不同时，可以观察到tradeMoney的走势相同，可知tradeMoney和houseFloor因素无关。
