在上一节中的<btn>[泰坦尼克号入门案例](https://www.liizhi.cn/2020/02/09/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E5%86%B3%E7%AD%96%E6%A0%91%E5%85%A5%E9%97%A8%E4%B9%8B%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7%E6%A1%88%E4%BE%8B/)</btn>的数据预处理过程中，出现了数据不完整、数据的编码(数值转化)，即将非结构化文本转化为结构化文本。本文主要用来记录在`sklearn`中常用的数据预处理基本方法。

### 数据预处理

从数据中检测，纠正或删除损坏，不准确或不适用于模型的记录的过程。

可能面对的问题有：数据类型不同，比如有的是文字，有的是数字，有的含时间序列，有的连续，有的间断。也可能，数据的质量不行，有噪声，有异常，有缺失，数据出错，量纲不一，有重复，数据是偏态，数据量太大或太小。

目的：让数据适应模型，匹配模型的需求。

#### 1. 数据无量纲化

在机器学习算法实践中，往往有着将不同规格的数据转换到同一规格，或不同分布的数据转换到某个特定分布的需求，这种需求统称为将数据“无量纲化”。 

数据的无量纲化包括线性与非线性。其中线性的无量纲化包括：**中心化**(Zero-centered或Mean-subtraction)处理和**缩放处理**(Scale)。

1. **中心化**

   让所有记录减去一个固定值，即让数据的样本数据平移到某个位置。

2. **缩放处理**

   通过除以一个固定值，将数据固定在某个范围之中，通常采用取对数的方式。

##### 1.1 数据归一化

当数据(x)按照最小值中心化后，再按极差（最大值-最小值）缩放，数据移动了最小值个单位，并且会被收敛到[0,1]之间，而这个过程，就叫做**数据归一化**(Normalization，又称Min-MaxScaling)。公式如下：

​																			$$x={x^*-min(x)\over max(x)-min(x)}$$

在`sklearn`中通过`preprocessing.MinMaxScaler`实现此功能。其中，`feature_range`可以控制数据压缩的范围，默认为[0,1]。

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
data = [[-1,2],[-0.5,6],[0,10],[1,18]]
pd.DataFrame(data)
# 实现归一化
scaler = MinMaxScaler() # 实例化
scaler = scaler.fit(data) # 生成min(x),max(x)
result = scaler.transform(data) # 导出结果
```

结果输出：

```python
array([[0.  , 0.  ],
       [0.25, 0.25],
	   [0.5 , 0.5 ],
       [1.  , 1.  ]])
```

将所有的数据压缩至[0,1]之间。

```python
scaler.inverse_transform(result) #逆转结果
Out:
	array([[-1. ,  2. ],
       	   [-0.5,  6. ],
           [ 0. , 10. ],
           [ 1. , 18. ]])
```

采用`feature_range`将数据范围压缩至[0,5]之间。

```python
# 使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
data = [[-1,2],[-0.5,6],[0,10],[1,18]]
scaler = MinMaxScaler(feature_range=[5,10]) # 实例化归一化到5~10之间
result = scaler.fit_transform(data)
result
Out：
    array([[ 5.  ,  5.  ],
           [ 6.25,  6.25],
           [ 7.5 ,  7.5 ],
           [10.  , 10.  ]])
```

采用`Numpy`实现归一化处理。

```python
# 使用numpy来实现归一化
import numpy as np
X = np.array(data)
X
Out:
    array([[-1. ,  2. ],
           [-0.5,  6. ],
           [ 0. , 10. ],
           [ 1. , 18. ]])
X_nor = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
X_nor
Out:
    array([[0.  , 0.  ],
           [0.25, 0.25],
           [0.5 , 0.5 ],
           [1.  , 1.  ]])
# 还原,即：公式的还原
X = X_nor * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
X
Out:
    array([[-1. ,  2. ],
           [-0.5,  6. ],
           [ 0. , 10. ],
           [ 1. , 18. ]])
```

通过以上的实例，将数据压缩至统一的范围内。

##### 1.2 数据标准化

当数据(x)按均值(μ)中心化后，再按标准差(σ)缩放，数据就会服从为均值为0，方差为1的正态分布（即标准正态分布），而这个过程，就叫做**数据标准化**(Standardization，又称Z-scorenormalization)，公式如下：

​																			$$x^*={x-u\over \sigma} $$

`sklearn`中提供了`preprocessing.StandarScaler`接口进行使用。

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # 实例化
scaler.fit(data) # fit,本质用于生成均值和方差
```

```python
# 对每一列向量表示一个特征，故默认对列进行操作
scaler.mean_ # 查看均值的属性mean_
scaler.var_  # 查看方差的属性var_
Out:
	array([-0.125,  9.   ])
```

导出结果：

```python
# 导出结果
x_std = scaler.transform(data)
x_std
Out:
    array([[-1.18321596, -1.18321596],
           [-0.50709255, -0.50709255],
           [ 0.16903085,  0.16903085],
           [ 1.52127766,  1.52127766]])
```

查看其方差与均值

```python
# 结果均值为0，方差为1的标准正太分布
x_std.mean()
x_std.std()
```

逆标准化

```python
scaler.inverse_transform(x_std) # 使用inverse_transform逆标准化
Out:
    array([[-1. ,  2. ],
           [-0.5,  6. ],
           [ 0. , 10. ],
           [ 1. , 18. ]])
```

##### 1.3 小结

**目的：**为了把不同来源的数据（不同特征）统一到同一数量级（一个参考坐标系）下，消除指标之间的量纲影响，解决数据指标简单可比性问题。

**优点：**

* 提高精度
* 可提高梯度下降求最优解的速度

#### 2. 数据缺失值的处理

此小节记录对于`sklearn`中缺失值处理的基本方法。

导入数据：

```python
import pandas as pd
data = pd.read_csv(r'/jupyter-notebook/sklearn/2- Feature Engineering/Narrativedata.csv',index_col=0)
data.head()
data.info()
Out:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 0 to 890
    Data columns (total 4 columns):
    Age         714 non-null float64
    Sex         891 non-null object
    Embarked    889 non-null object
    Survived    891 non-null object
    dtypes: float64(1), object(3)
    memory usage: 34.8+ KB
```

从以上结果中可以看出，共有891条数据，其中`Age`,`Embarked`皆存在缺失值。`sklearn`中提供了`sklearn.impute.SimpleImputer`接口处理缺失值。

首先对Age缺失值处理方式：

```python
from sklearn.impute import SimpleImputer
Age = data.loc[:,'Age'].values.reshape(-1,1)
imp_mean = SimpleImputer() # 实例化,默认均值填补
imp_median = SimpleImputer(strategy='median') # 采取中位数填补
imp_0 = SimpleImputer(strategy='constant',fill_value=0) # 给定常数，以0填补
imp_most = SimpleImputer(strategy='most_frequent')#采用众数进行填补，可用于字符串
imp_mean = imp_mean.fit(Age)
imp_mean = imp_mean.transform(Age)
imp_median = imp_median.fit_transform(Age)
imp_most = imp_most.fit_transform(Age)
```

结果输出，取前5个数据。

```python
imp_mean[:5]
imp_median[:5]
imp_most[:5]
Out: # 采用众数进行填补的结果
    array([[22.],
           [38.],
           [26.],
           [35.],
           [35.]])
```

将众数作为`Age`缺失值处理的方式：

```python
Age = imp_most
data.loc[:,'Age'] = Age
```

对`Embarked`处理的方式：

```python
# 采用众数填补Embarked
Embarked = data.loc[:,'Embarked'].values.reshape(-1,1)
imp_most = SimpleImputer(strategy='most_frequent')
imp_most = imp_most.fit_transform(Embarked)
Embarked = imp_most
data.loc[:,'Embarked'] = Embarked
```

**注意：**众数的施加对象可以是非数值型。

**补充：**

采用`Pandas`和`Numpy`进行缺失值的填补

```python
# 采用平均值填补年龄的缺失值,利用.fillna 在DataFrame里面进行填补
data_.loc[:,'Age'] = data_.loc[:,'Age'].fillna(data.loc[:,'Age'].mean())
# 删除Embarked缺失的两条记录,dropna(axis=0)删除所有有缺失值的行，.dropna(axis=1) 删除所有有缺失值的列
# 当采用删除操作时axis=0是对行操作，axis=1是对列操作；拼接，切片相反
data_.dropna(axis=0,inplace=True)
data_.info()
Out:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 889 entries, 0 to 890
    Data columns (total 4 columns):
    Age         889 non-null float64
    Sex         889 non-null object
    Embarked    889 non-null object
    Survived    889 non-null object
    dtypes: float64(1), object(3)
    memory usage: 34.7+ KB
```

#### 3. 编码与哑变量

在机器学习中，大多数算法，譬如逻辑回归，支持向量机SVM，k近邻算法等都只能够处理数值型数据，不能处理文字，在sklearn当中，除了专用来处理文字的算法，其他算法在ﬁt的时候全部要求输入数组或矩阵，也不能够导入文字型数据（其实手写决策树和普斯贝叶斯可以处理文字，但是sklearn中规定必须导入数值型）。
然而在现实中，许多标签和特征在数据收集完毕的时候，都不是以数字来表现的。比如说，学历的取值可以是["小学"，“初中”，“高中”，"大学"]，付费方式可能包含["支付宝"，“现金”，“微信”]等等。在这种情况下，为了让数据适应算法和库，我们必须将数据进行编码，即是说，将文字型数据转换为数值型。

##### 3.1 标签的编码

`preprocessing.LabelEncoder`:标签专用，能够将分类转换为分类数值

```python
from sklearn.preprocessing import LabelEncoder # 对标签进行编码
y = data.iloc[:,-1] # 取出特征，最后一列，标签允许是一维
le = LabelEncoder()  # 实例化
le = le.fit(y)
label = le.transform(y)
data.iloc[:,-1] = label
le.classes_ # 查看标签中类别数量
Out:
	array(['No', 'Unknown', 'Yes'], dtype=object)
```

查看标签`Survived`这一列：

```python
# 取前5条数据查看
data['Survived'][:5]
Out:
    0    0
    1    2
    2    2
    3    2
    4    0
    Name: Survived, dtype: int64
```

##### 3.2 特征的编码

`preprocessing.OrdinalEncoder`:特征专用，能够将分类特征转换为分类数值。

```python
fromsklearn.preprocessingimportOrdinalEncoder 
#接口categories_对应LabelEncoder的接口classes_，一模一样的功能
data_=data.copy()
data_.head()
OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_
data_.iloc[:,1:-1]=OrdinalEncoder().fit_transform(data_.iloc[:,1:-1])
data_.head()
```

##### 3.3 独热编码——创建哑变量

类别OrdinalEncoder可以用来处理有序变量，但对于名义变量，我们只有使用哑变量的方式来处理，才能够尽量向算法传达最准确的信息。

- 名义变量

  判断两变量是否相同。例如：性别，邮编，身份证号等等

- 有序变量

  为数据的相对大小提供信息，但数据之间大小的间隔不是具有固定意义的，不能做加减运算。例如：学历。

- 有距变量

  有距变量之间的间隔是有固定意义的，可做加减运算。例如：日期

从以上定义看出，性别、舱门号等属于有序变量。在之前的编码中，性别简单采用的`0\1`区别`男\女`。但是，在编码的过程中，想要表达的是`男≠女`。当被我们转换为`[0,1]`时，存在着大小关系，即从名义变量的编码转化成为了有距变量的编码。

故：我们采用独热编码(one-hot)的形式进行编码。男:[1,0],女:[0,1]。这样，便能够将男女的编码区别于一般的0、1编码，让算法明白这两取值是没有计算性质的，这种编码即为哑变量。

在`sklearn`中提供了`sklearn.preprocessing.OneHotEncoder`接口进行哑变量处理。

```python
from sklearn.preprocessing import OneHotEncoder
X = data.iloc[:,1:-1] #取特征,即：Sex、Embarked

# one-hot
enc = OneHotEncoder() # 实例化
enc = enc.fit(X)
result = enc.transform(X)
result
Out:
    <889x5 sparse matrix of type '<class 'numpy.float64'>'
        with 1778 stored elements in Compressed Sparse Row format>
```

`result`中返回的是结果集对象地址。

```python
result.toarray()
Out:
    array([[0., 1., 0., 0., 1.],
           [1., 0., 1., 0., 0.],
           [1., 0., 0., 0., 1.],
           ...,
           [1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0.],
           [0., 1., 0., 1., 0.]])
```

从结果中，看出我们得到5列特征。其中，Age包含男女两类，Embarked包含S、Q、C三类。故通过One-hot得到了5类特征。

```python
enc.get_feature_names() # 用于查看特征默认的名称
Out:
    array(['x0_female', 'x0_male', 'x1_C', 'x1_Q', 'x1_S'], dtype=object)
```

将新得到的特征表示，拼接至原有数据后：

```python
newdata = pd.concat([data,pd.DataFrame(result)],axis=1)# 将数据进行拼接
newdata.drop(['Sex','Embarked'],inplace=True,axis=1) # 删除原来的特征
newdata.columns = ['Age','Survived','Female','Male','Embarked_C','Embarked_Q','Embarked_S'] # 列名重命名
```

#### 4. 连续型特征处理：二值化与分段

在上一小节的特征处理中，one-hot处理的是离散型变量。根据阈值将数据二值化（将特征值设置为0或1），用于处理连续型变量。大于阈值的值映射为1，而小于或等于阈值的值映射为0。默认阈值为0时，特征中所有的正值都映射到1。

二值化是对文本计数数据的常见操作，分析人员可以决定仅考虑某种现象的存在与否。它还可以用作考虑布尔随机变量的估计器的预处理步骤（例如，使用贝叶斯设置中的伯努利分布建模）。

`sklearn`中提供了`sklearn.preprocessing.Binarizer`用于连续型数据的二值化处理。

```python
from sklearn.preprocessing import Binarizer # 用于将根阈值将数据二值化，处理连续型变量的工具包
data_2 = data.copy()
X = data_2.iloc[:,0].values.reshape(-1,1)
transformer = Binarizer(threshold=30).fit_transform(X) # threshold=30，即以30作为二值化分段的界限
transformer[:4]
Out:
    array([[0],
           [1],
           [0],
           [1]])
```

从年龄结果的前4条数据看出，年龄大于30的映射为1，小于等于30的映射为0。

`sklearn.preprocessing.KBinsDiscretizer`可用于设计连续型变量数据的n分类。

参数解释：

|   参数   | 含义&输入                                                    |
| :------: | :----------------------------------------------------------- |
|  n_bins  | 每个特征中分箱的个数，默认5，一次会被运用到所有导入的特征    |
|  ncode   | 编码的方式，默认“onehot”<br/>"onehot"：做哑变量，之后返回一个稀疏矩阵，每一列是一个特征中的一个类别，含有该类别的样本表示为1，不含的表示为0 <br/>“ordinal”：每个特征的每个箱都被编码为一个整数，返回每一列是一个特征，每个特征下含有不同整数编码的箱的矩阵<br/>"onehot-dense"：做哑变量，之后返回一个密集数组。 |
| strategy | 用来定义箱宽的方式，默认"quantile"<br/>"uniform"：表示等宽分箱，即每个特征中的每个箱的最大值之间的差为(特征.max()-特征.min())/(n_bins)"<br/>quantile"：表示等位分箱，即每个特征中的每个箱内的样本数量都相同<br/>"kmeans"：表示按聚类分箱，每个箱中的值到最近的一维k均值聚类的簇心得距离都相同 |



```python
from sklearn.preprocessing import KBinsDiscretizer
X = data.iloc[:,0].values.reshape(-1,1)
# n_bins 为划分的数量，即需要划分多少类。
est = KBinsDiscretizer(n_bins=6,encode='ordinal',strategy='uniform')
t = est.fit_transform(X)
t[:10]
Out:
    array([[1.],
           [2.],
           [1.],
           [2.],
           [2.],
           [2.],
           [4.],
           [0.],
           [2.],
           [1.]])
set(t.ravel()) # .ravel() 用于降维，set集合去重，查看类别的数量
```



