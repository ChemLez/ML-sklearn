<font size=4>本篇用于记录机器学习中的一次入门练习，即：利用决策树进行简单的二分类。同时，结合Kaggle上的经典案例Titanic，来测试实际效果。

# 一、数据集
采用[Kaggle](https://www.kaggle.com/c/titanic/data)中的Titanic的数据集。数据包含分为：  
- 训练集: training set (train.csv)
- 测试集: test set (test.csv)
- 提交标准: gender_submission.csv  
  

由于Kaggle涉及到科学上网的操作，可点击[原始数据集](https://github.com/ChemLez/ML-sklearn/tree/master/1-%20DecisionTree)在我的仓库中下载。
# 二、数据处理
首先导入训练集，查看数据的情况：  
```python
from sklearn.tree import DecisionTreeClassifier # 导入模型决策树分类器
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV # 导入的模型作用分别为交叉验证、训练集与数据集的划分，网格搜索
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/liz/code/jupyter-notebook/sklearn/1- DecisionTree/Titanic_train.csv') # 导入数据集
data.head() # 显示数据集的前五行
[out]:
```
<!--more-->  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>


通过以上的数据所展示的情况，我们所要做的是将Survived作为标签，其余的列作为特征。目标：以所知的特征来预测标签。这份数据集的实际意义是:通过已知数据对乘客的生还情况做一次预测。
```python
data.info() # 查看整个训练集的情况
out:
 <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
```
##### 数据分析  

1. 通过以上的数据展示，共有891条数据，其中具有缺失值的特征有：Age、Cabin、Embarked；非数值型的特征有：Name,Sex,Ticket,Cabin,Embarked。
2. 当我们采用现有的特征对乘客进行生还情况预测时，一些处理较为麻烦且不太重要的特征对可不采用。例如：这里的Name、Ticket可以不采用，因为在实际情况中乘客的名字以及所购的票对于乘客的生还情况作用不大。另外一点原因是这两者皆为非数值型数据，处理成数值形式较为复杂（在计算机中所接受的数据最终都要以数字的形式进行呈现）。
3. 由于Cabin缺失值较多，这里采用删除的方式，理由同上。
4. 虽然性别也为字符型数据，当在实际中性别对于逃生的可能性具有一定的影响，故对其保留。
5. 将缺失值进行填补；将非数值型数据转化为数值型数据。    
```python
# 删除Name、Ticket、Cabin特征列
data.drop(['Name','Cabin','Ticket'],inplace=True,axis=1)


# 缺失值的填补
# 对于Age的缺失值填补的一种策略为：以年龄的平均值作为填补
data.loc[:,'Age'] = data['Age'].fillna(int(data['Age'].mean()))
# Embarked由于只有两条数据具有缺失值，这里采用的方式是删除这两条缺失的数据（缺失两条数据对模型的训练好坏影响不大）
data = data.dropna()
data = data.reset_index(drop = True) # 删除过后，用于重置索引

# 将非数值型数据转化为数值型数据
# 性别只有两类，故可用0\1来表示男女
data['Sex'] = (data['Sex'] == 'male').astype(int) # 0表示女，1表示男
tags = data['Embarked'].unique().tolist() # tags: ['S', 'C', 'Q']
# Embarked只有三类分别以S,C,Q的索引代表他们,0~9均可采用此种方法
data.iloc[:,data.columns == 'Embarked'] = data['Embarked'].apply(lambda x : tags.index(x))

# 查看数据
data.info() # 查看数据信息
out:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 889 entries, 0 to 888
Data columns (total 9 columns):
PassengerId    889 non-null int64
Survived       889 non-null int64
Pclass         889 non-null int64
Sex            889 non-null int64
Age            889 non-null float64
SibSp          889 non-null int64
Parch          889 non-null int64
Fare           889 non-null float64
Embarked       889 non-null int64
dtypes: float64(2), int64(7)
memory usage: 62.6 KB


# 将特征与标签进行分离
x = data.iloc[:,data.columns != 'Survived'] # 取出Survived以为的列作为特征x
y = data.iloc[:,data.columns == 'Survived'] # 取出Survived列作为特征y
```

##### 模型训练

思路：采用交叉验证来评估我们的模型；同时采用网格搜索来查找决策树中常见的最佳参数。

```python
# 网格搜索：能够帮助我们同时调整多个参数的技术，本质是枚举技术。
# paramerters：用于确定的参数。
parameters = {'splitter':('best','random')
             ,'criterion':('gini','entropy')
             ,'max_depth':[*range(1,10)]
             ,'min_samples_leaf':[*range(1,50,5)]
             ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
             }

# 网格搜索实例代码，所需要确定的参数越多，耗时越长
clf = DecisionTreeClassifier(random_state=30)
GS = GridSearchCV(clf,parameters,cv=10) # cv=10,做10次交叉验证
GS = GS.fit(x_train,y_train)

# 最佳参数
GS.best_params_
out:
    {'criterion': 'gini',
 'max_depth': 3,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'splitter': 'best'}
    
# 最佳得分
GS.best_score_
```

确定了设置的参数的最佳值，开始训练模型：

```python
# 训练模型，将以上设置参数的最佳值填入模型的实例化中
clf_model = DecisionTreeClassifier(criterion='gini'
                                  ,max_depth=3
                                  ,min_samples_leaf=1
                                  ,min_impurity_decrease=0
                                  ,splitter='best'
                                  )
clf_model = clf_model.fit(x,y)
```

导出模型：

```python
# 导出模型
from sklearn.externals import joblib
joblib.dump(clf_model,'/Users/liz/Code/jupyter-notebook/sklearn/1- DecisionTree/clf_model.m')
```

测试集的处理：

```python
# 导入测试集
data_test = pd.read_csv('/Users/liz/code/jupyter-notebook/sklearn/1- DecisionTree/Titanic_test.csv')
data_test.info()
out:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB

# 测试集处理的方法同训练集，同时测试集要与训练集保持同样的特征
# 由于最后，我们需要将处理结果上传到Kaggle上，所以不能够将数据条目减少，即：需要上传418条测试数据；故这里Fare缺失的一条数目同样采用平均值来填补
data_test.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)
data_test['Age'] = data_test['Age'].fillna(int(data_test['Age'].mean()))
data_test['Fare'] = data_test['Fare'].fillna(int(data_test['Fare'].mean()))
data_test.loc[:,'Sex'] = (data_test['Sex'] == 'male').astype(int)
tags = data_test['Embarked'].unique().tolist()
data_test['Embarked'] = data_test['Embarked'].apply(lambda x : tags.index(x))
```

此时测试集数据预处理完毕，导出模型并对数据进行测试：

```python
# 导出模型且测试数据集
model = joblib.load('/Users/liz/Code/jupyter-notebook/sklearn/1- DecisionTree/clf_model.m')
Survived = model.predict(data_test) # 测试结果
# 生成数据
Survived = pd.DataFrame({'Survived':Survived}) # 将结果转换为字典形式并后续作为csv形式导出
PassengerId = data_test.iloc[:,data_test.columns == 'PassengerId'] # 切片，分割出PassengerId
gender_submission = pd.concat([PassengerId,Survived],axis=1)# 将Survived与PassengerId拼接，一一对应

#导出数据
#导出数据
gender_submission.index = np.arange(1, len(gender_submission)+1) # 索引从1开始
gender_submission.to_csv('/Users/liz/Code/jupyter-notebook/sklearn/1- DecisionTree/gender_submission.csv',index=False) # index=False，导出时不显示索引
```

导出文件:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>896</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>413</td>
      <td>1305</td>
      <td>0</td>
    </tr>
    <tr>
      <td>414</td>
      <td>1306</td>
      <td>1</td>
    </tr>
    <tr>
      <td>415</td>
      <td>1307</td>
      <td>0</td>
    </tr>
    <tr>
      <td>416</td>
      <td>1308</td>
      <td>0</td>
    </tr>
    <tr>
      <td>417</td>
      <td>1309</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 2 columns</p>

将结果提交到[Kaggle](https://www.kaggle.com/c/titanic)上，最终得分：

![得分](https://s2.ax1x.com/2020/02/09/1WxMnK.png)

最终得分0.77990。此篇只是作为机器学习及Kaggle的一个入门，完整的过一遍流程。

