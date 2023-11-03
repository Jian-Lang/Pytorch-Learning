# **Pytorch进阶实战**

```html
<p>
    One day, you will do things that others can not do!
</p>
```

## Part1: Pytorch创建数据集通用模板

### 1. 模板展示

```python
import torch.utils.data
import pandas as pd


def custom_collate_fn(batch):

    inputs, targets = zip(*batch)

    return torch.tensor(inputs), torch.tensor(targets)

class MyData(torch.utils.data.Dataset):

    def __init__(self,path):
        super().__init__()
        self.path = path
        self.csv = pd.read_csv(self.path)

    def __getitem__(self, item):

        input_ = [float(self.csv['weight'][item]),float(self.csv['height'][item])]
        target = float(self.csv['sex'][item])
        return (input_, target)

    def __len__(self):
        return len(self.csv['weight'])
    
```

------



### 2. 模板说明

(1) 模板中仅包含训练(train)数据集的loader，test和valid同理;

(2) collate_fn表示需要把数据打包成什么样的类型；

(3) 理论上，传给DataLoader()的dataset需要具备**三个函数**：

初始化函数__init__()

获取元素__getitem()__

获取长度__len__（）

这三个函数具备之后，即可传入，其中getitem函数需要返回完整的一组数据(返回时可按tuple类型)。上面的例子是对csv数据的处理。

------



### 3. 数据集划分规则

一般而言，**对于原始数据集，要划分为8 ：1：1的三部分**，分别作为**train、test和valid**的数据使用，其中train和test在训练模型

时可见，但valid数据集应当始终为模型不可见，直到通过test拿到好的model后，再在valid上进行验证。

------



## Part2: Pytorch定义模型通用模板

### 1.模板展示

```python
import torch.nn

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            xxx
        )
    
    def forward(self,x):
        return self.model(x)
```

------



### 2.注意事项

一定要注意，**所有模型的参数，都要定义在init构造函数内，不要把模型参数定义在forward中**，forward只负责输出结果，正确的做法是在构造函数内部定义相应的处理过程(例如 Relu，pooling，Linear等等)，而后在forward中进行输出。

------



## Part3: Pytorch训练模型通用模板

### 1. 模板展示

```python
import torch
import os
import logging
from torch.utils.data import DataLoader
from xxx import Model
from datetime import datetime
from tqdm import tqdm

if __name__ == "__main__":

    # dataset id

    dataset_id = "xxx"

    # metric

    metric = "xxx"

    # 创建文件夹和日志文件，用于记录训练结果和模型

    # 获取当前时间戳

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件夹名称
    folder_name = f"train_{dataset_id}_{metric}_{timestamp}"

    # 这里注意，要创建一个文件夹叫train_results，否则会报错

    # 指定文件夹的完整路径

    folder_path = os.path.join("train_results", folder_name)

    # 创建文件夹

    os.mkdir(folder_path)

    os.mkdir(os.path.join(folder_path, "trained_model"))

    # 配置日志记录

    # 创建logger对象

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    # 创建控制台处理器

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    # 创建文件处理器

    file_handler = logging.FileHandler(f'train_results/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    # 设置日志格式

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    # 将处理器添加到logger对象中

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    # 加载数据集

    train_data = xxx

    test_data = xxx

    batch_size = 256

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, collate_fn=xxx)

    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=xxx)

    # 加载待训练的模型

    model = Model(xxx)

    model = model.to('cuda:0')

    # 定义损失函数

    loss_fn = torch.nn.xxxfunction()

    loss_fn.to('cuda:0')

    # 定义优化器

    learning_rate = 0.001

    optim = torch.optim.Adam(model.parameters(), learning_rate)

    # 设置max_epoch,并在超过n轮没有好转的情况下，直接结束训练

    max_epoch = 1000

    max_no_optim_turn = 20

    # 定义训练过程的一些参数

    total_train_step = 0

    min_total_test_loss = 1008611

    min_turn = 0

    loss = 0

    # 训练部分

    for i in range(max_epoch):

        logger.info(f"----------Epoch {i + 1} Start!----------")

        # 训练环节

        model.train()
        
        min_train_loss = 1008611

        for batch in tqdm(train_data_loader,desc='Training Progress'):

            input_, target = batch

            input_ = input_.to('cuda:0')

            target = target.to('cuda:0')

            output = model.forward(input_)

            loss = loss_fn(output, target)

            # 通过损失，优化参数

            optim.zero_grad()

            loss.backward()

            optim.step()

            total_train_step += 1
            
            if min_train_loss > loss:

                min_train_loss = loss

        logger.info(f"[ Epoch {i + 1} (train) ]: loss is {min_train_loss}")

        # 测试环节

        model.eval()

        total_test_loss = 0

        with torch.no_grad():

            for batch in tqdm(test_data_loader,desc='Testing Progress'):

                input_, target = batch

                input_ = input_.to('cuda:0')

                target = target.to('cuda:0')

                output = model.forward(input_)

                loss = loss_fn(output, target)

                total_test_loss += loss

        if total_test_loss < min_total_test_loss:

            min_total_test_loss = total_test_loss

            min_turn = i + 1

        logger.info(f"[ Epoch {i + 1} (test) ]: total_loss is {total_test_loss}")

        logger.critical(f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_test_loss}")

        torch.save(model, f"train_results/{folder_name}/trained_model/model_{i + 1}.pth")

        logger.info("Model has been saved successfully!")

        if (i + 1) - min_turn > max_no_optim_turn:

            break

# 删除掉垃圾模型

model_name_list = os.listdir(f"train_results/{folder_name}/trained_model")

for i in range(len(model_name_list)):

    if model_name_list[i] != f'model_{min_turn}.pth':

        os.remove(os.path.join(f'train_results/{folder_name}/trained_model', model_name_list[i]))

logger.info("Training is ended!")
```

------



### 2. 细节补充说明

#### 2.1 使用GPU加速

当训练模型时，需要使用gpu加速，可以使用以下两种形式：

```python
# 方式一

xxx.to('cuda:x')

# 方式二

device = torch.device('cuda:x')

xxx.to(device)
```

需要进行此操作的对象包括：**模型、损失函数和数据**：

```python
model = model.to('cuda:x')

batch = batch.to('cuda:x')

loss_fn = loss_fn.to('cuda:x')
```

------



#### 2.2 使用tensorboard可视化训练过程





## Part4: Pytorch验证模型通用模板

### 1. 模板展示

```python
import os
from datetime import datetime
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


if __name__ == "__main__":

    # dataset id

    dataset_id = "xxx"

    # metric

    metric = "xxx"

    # 创建文件夹和日志文件，用于记录验证集的结果

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 定义文件夹名称
    folder_name = f"valid_{dataset_id}_{metric}_{timestamp}"

    # 指定文件夹的完整路径
    folder_path = os.path.join("valid_results", folder_name)

    # 创建文件夹
    os.mkdir(folder_path)

    # 配置日志记录

    # 创建logger对象

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    # 创建控制台处理器

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    # 创建文件处理器

    file_handler = logging.FileHandler(f'valid_results/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    # 设置日志格式

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    # 将处理器添加到logger对象中

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    # 加载数据集

    batch_size = 64

    valid_data = xxx

    valid_data_loader = DataLoader(dataset=valid_data,batch_size=batch_size,drop_last=True,collate_fn=xxx)

    # 加载训练好的模型, 这里记得import你的模型

    model = torch.load(r'xxx')

    # 定义损失函数

    loss_fn = torch.nn.BCELoss()

    loss_fn.to('cuda:0')

    # 定义验证相关参数

    total_valid_step = 1

    # 开始验证

    model.eval()

    with torch.no_grad():

        logger.info("Validation starts!")

        for batch in tqdm(valid_data_loader,desc='Validating'):

            input_,target = batch

            input_ = input_.to('cuda:0')

            target = target.to('cuda:0')

            target = target.view(-1,1)

            output = model.forward(input_)

            loss = loss_fn(output,target)

            # logger.warning(f"[ Batch {total_valid_step} (valid) ]:  loss = {loss}")

            total_valid_step += 1

    logger.info("Validation is ended!")
```

------



### 2. 细节补充说明

在验证时，一定要先在**import处导入原模型的定义**，而后再使用相应的两种方式加载模型：

```python
# 1.torch.load()

model = torch.load(path)

# 2.torch.load_state_dict

# 此处model需要提前被通过正确的方式初始化，而后才可以加载模型，例如model = Model()

model.load_state_dict(torch.load(path))
```

第一种方法，对应保存的是完整模型；

第二种方法，对应保存的仅仅是模型参数，**此时注意，model需要先被初始化**，初始化之后，才能调用load_state_dict()方法，并传入torch.load()方法把参数load进去。

------



## Part5: 数据集划分通用模板

```python
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    initial_data_frame = pd.read_pickle(r'')

    # 下面的部分, 编写对原始数据集dataframe的各种修改即可

    # 生成训练集、测试集和验证集

    train_data_frame, others_data_frame = train_test_split(initial_data_frame, test_size=0.2, random_state=7)

    test_data_frame, valid_data_frame = train_test_split(others_data_frame, test_size=0.5, random_state=7)

    # 修正索引

    train_data_frame.reset_index(drop=True, inplace=True)

    test_data_frame.reset_index(drop=True, inplace=True)

    valid_data_frame.reset_index(drop=True, inplace=True)

    # 保存为pickle文件

    train_data_frame.to_pickle(r'')

    test_data_frame.to_pickle(r'')

    valid_data_frame.to_pickle(r'')

```



------



## Part6: Pytorch常用函数

### 1.torch.linspace()



### 2.torch.stack()



### 3.torch.empty()



### 



### 5.view()



### 6.squeeze()与unsqueeze()



### 7.torch.cat



### 8.size(), shape



### 9.张量乘法相关函数



### 10.转置相关操作



### 11.类型相关函数

查询类型，修改类型



## Part7: Pytorch重要知识点笔记

### 1.张量之间的各种乘法与数乘

#### 1.1 使用*做乘法

#### 1.2 mm函数

#### 1.3 mul函数

#### 1.4 matmul函数

### 2.张量转置操作

#### 2.1普通转置



#### 2.2指定维度转置



### 3.张量各种拼接操作

#### 3.1cat函数

#### 3.2stack函数

### 4.张量尺寸及尺寸变化

#### 4.1查看尺寸的方法



#### 4.2尺寸修改



#### 4.3扩大或缩小一维



### 5.张量类型及转换

#### 5.1查看张量类型



#### 5.2张量类型转换



## Part8: sklearn常用函数速查

## Part9: Python数据处理(Additional)

### Numpy

#### 1.导入Numpy

使用下面这行代码导入Numpy，并命名为np：

```python
import Numpy as np
```

------



#### 2.创建Numpy数组

##### 2.1 通过列表创建Numpy数组

```python
a = np.array([1,2,3,4],dtype = 'float')
```

这行代码生成了一个Numpy类型数组，并指定数据元素的类型为float。dtype可以省略。

要注意的是，在numpy数组中，**不允许有两种数据类型**，要求全部元素同一个类型！

------



##### 2.2 两种特殊的Numpy数组

生成全0的Numpy数组：

```python
a = np.zeros(5,dtype = 'float')
```

传入的参数表示生成的数组的尺寸，这里的尺寸可以是一维的，也可以是高维的。

生成全1的Numpy数组：

```python
a = np.ones((2,3),dtype = 'float')

# a = [[0,0,0],[0,0,0]]
```

**注意，高维的时候，传入的多个维度的尺寸要放在一个圆括号当中！**

------



##### 2.3 生成序列型Numpy数组

```python
a = np.arange(1,10)

# a = [0 1 2 3 4 5 6 7 8 9]
```

注意，这里是左闭右开，如果只有一个数字，默认数组元素是从0 到 这个数字 - 1

------



##### 2.4 生成随机数Numpy数组

生成 0 - 1范围内的随机数Numpy数组：

```python
a = np.random.rand(10)
```

同样的，括号内的参数代表尺寸，支持多维度，多维度时，**直接传入多个维度的尺寸即可，无需用圆括号！**

生成**正态分布**的随机数Numpy数组：

```python
a = np.random.randn(10)
```

在rand后面加了一个n。

生成整型的随机数：

```python
a = np.random.randint(1,10,10)
```

上面代码生成1 - 10范围内的10个随机数，最后一位代表尺寸，同样支持多维度。

------



##### 2.5 生成采样Numpy数组

```python
a = np.linspace(1,10,100)
```

生成1 - 10范围内的100个采样，注意，**这些采样不是随机的，是按照等间距进行的采样**，并且采样的结果按照从小到大排列。

------



##### 2.6 用某个值填充Numpy数组

```python
a = np.array([1,2,3,4,5])

a.fill(6)

# a = [6,6,6,6,6]
```

以fill()中的值完全填充原来的Numpy数组。

------



#### 3.Numpy数组的常用运算

##### 3.1 单个数组每个元素的运算

数组每个元素加上某个值：

```python
a = np.array([1,2,3,4,5])

a = a + 1

# a = [2,3,4,5,6]
```

数组每个元素乘多少倍：

```python
a = np.array([1,2,3,4,5])

a = a * 2

# a = [ 2  4  6  8 10]
```

总结：对每个元素要进行什么样的操作，在numpy数组当中，只需要对数组变量进行该操作即可。

------



##### 3.2 两个数组之间的运算

两个数组的每个元素相加：

```python
a = np.array([1,2,3,4,5])

b = np.array([2,3,4,5,6])

c = a + b

# c = [ 3  5  7  9 11]
```

两个数组的每个元素相乘：

```python
a = np.array([1,2,3,4,5])

b = np.array([2,3,4,5,6])

c = a * b

# c = [ 2  6 12 20 30]
```

总结：同3.1

------



#### 4.Numpy数据元素类型与尺寸

##### 4.1 类型查看

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a.dtype) # float

print(type(a)) # ndarray
```

注意，上面两个并不相同，第一个是a内的数据元素的类型，第二个是a本身的类型。

------



##### 4.2 类型转换

```python
a = np.array([1,2,3,4,5],dtype = 'float')

a = a.astype('int')

print(a.dtype) # int
```

使用astype函数即可完成类型的转换。

------



##### 4.3 尺寸查看

```python
a = np.array([[1,2,3],[4,5,6]])

print(a.shape) # (2,3)

b = np.array([[[1,2,3],[4,5,6]]]) # (1,2,3)

# 某一维的尺寸查看：

len(a)  # 第0维

len(a[0]) # 第一维
```

这里注意，一般来说，无论是哪一种包，高维的数组当中，以三维为例：

第一维代表个数，即有多少个二维的数目，第二维代表行，第三维代表列。

------



##### 4.4 尺寸转换

对于确定的numpy数组，可以转换其尺寸，但是要遵循元素个数不变原则：

```python
a = np.array([1,2,3,4,5,6],dtype = 'float')

a = a.reshape(2,3)

# a = [[1. 2. 3.],[4. 5. 6.]]
```

注意**reshape函数并不会直接改变a本身**，而是生成一个新的数组，因此需要赋值给其他变量。

------



#### 5.Numpy数组索引与切片

##### 5.1 索引

Numpy数组的索引和普通的列表完全相同：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[0]) # 1.0
```

但是这里复习一下**负索引值**的使用：

注意，这里要这么理解：

在正向索引时，**index = 0表示数组的首位元素的索引**，于是在**反向索引**时，我们规定 **index = -1表示数组的最后一位元素的索引**：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[-1]) # 5.0
```

之后，倒着往前走的过程中，index的绝对值继续增加即可，例如index = -2表示数组倒数第二位元素的索引：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[-2]) # 4.0
```

------



##### 5.2 切片

Numpy数组切片与普通列表也基本相同：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[0:-1:2]) # [1 3]
```

[**begin index : end index : step**]

只是要注意的是，它是一个**左闭右开**的区间，就是说切出来的结果是begin --- end - 1

这里复习一下切片的缺省情况：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[1:]) # [2 3 4 5]
```

这是缺end index，此时会**默认从begin index取到最后一个元素**(最后一个元素也取)。

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[:-1]) # [1 2 3 4]
```

这是缺begin index，此时会**默认从第一个元素取到end index - 1的元素**。

缺步长时，默认为1，这个是很显然的。

再复习一下步长为负值的情况：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[::-1]) # [5. 4. 3. 2. 1.]
```

当步长为负值，则默认从end index - 1开始，向begin index开始取值，这里由于begin和end都缺少，表示整个数组倒着取值，完成了数组逆序。

------



##### 5.3 花式索引

与传统的索引不同，花式索引的意思是一次性传入多个索引，此时数组会同时把多个索引的值都返回，返回时返回的是一个numpy数组的形式：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

print(a[[0,1,2]]) # [1. 2. 3.]
```

注意，**多个索引要用一个中括号传入**，即[index1,index2,...]，当然，也**可以直接传进去一个list或一个numpy 数组**：

```python
a = np.array([1,2,3,4,5],dtype = 'float')

b = np.array([0,1,2])

print(a[b]) # [1. 2. 3.]
```

------



#### 6.Numpy高维数组

##### 6.1 高维数组的生成

首先最原始的方法肯定是通过：

```python
a = np.array([[1,2,3],[4,5,6]],dtype = 'float')
```

其他的方法，在之前第二部分也提到过，所有的那些方法基本上都支持生成高维的数组。

------



##### 6.2 高维数组的索引

```python
a = np.array([[1,2,3],[4,5,6]],dtype = 'float')

print(a[0,1]) # 2
```

对于高维数组来说，需要指定每个维度的索引值，每个维度之间以逗号分隔：a[index1,index2,index3...]

但是如果仅仅传入某一维度，则表示另一维度全部都取到，从而形成一个新的维度的数组：

```python
a = np.array([[1,2,3],[4,5,6]],dtype = 'float')

print(a[0]) # [1,2,3]
```

------



##### 6.3 高维数组的切片

高维numpy数组的切片，与低维度不同，切片时，需要对多个维度分别切片：

```python
a = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype = 'float')

print(a[0:2,:-1]) # [[1,2],[4,5]]
```

语法：a[维度1切片, 维度2切片,...]，即多个维度的切片以逗号分隔，每个维度的切片语法与一维数组切片完全相同。

特别地，如果没有对某个维度切片，则可以直接用一个:带过，或者直接省略。

------



##### 6.4 高维数组的花式索引

高维数组的花式索引，其实就是在一维数组的基础上，每一个维度都分别传入多个索引值，每个维度的多个索引值直接用逗号隔开：

```python
a = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype = 'float')

print(a[[0,1],[2,2]]) # [3 6]
```

要注意，每个维度的多个索引值，要和其他维度的索引值搭配起来，才能锁定元素，同样的，每个维度的索引值用方括号包起来：

a[[第一维度的多个索引],[第二维度的多个索引],..]

因此a[[0,1], [2,2]] 表示(0,2) 和(1,2)两个元素。

------



#### 7.Numpy常用函数

##### 7.1 指数函数

```python
a = np.array([1,2,3],dtype = 'float')

print(np.exp(a)) # [ 2.71828183  7.3890561  20.08553692]
```

这个函数表示把numpy数组的**每个元素**，都放在exp指数函数当中进行运算**e^x**，而后**返回同样尺寸的一个新数组**。

------



##### 7.2 最值函数

```python
a = np.array([1,2,3],dtype = 'float')

print(a.max()) # 3

print(a.min()) # 1
```

------



##### 7.3 求和函数

基本形式：

```python
a = np.array([1,2,3],dtype = 'float')

print(a.sum()) # 6
```

高维数组，指定维数求和：

```python
a = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype = 'float')

print(a.sum(axis = 0)) # [12,15,18]

print(a.sum(axis = 1)) # [6,15,24]
```

其中，**axis = 0 代表沿着纵向**，即**y轴方向**求和，**axis = 1 代表沿着横向**，即**x轴方向**求和。下面的数组拼接也用到了axis坐标轴这一概念。

------



##### 7.4 与统计相关的函数

```python
a = np.array([1,2,3],dtype = 'float')
```

计算平均值

```python
print(np.mean(a))
```

计算标准差

```python
print(np.std(a))
```

计算中位数

```python
print(np.median(a))
```

------



##### 7.5 矩阵转置

```python
a = np.array([[1,2,3],[4,5,6]],dtype = 'float')

a = a.T

print(a) # a = [[1. 4.][2. 5.][3. 6.]]
```

------



##### 7.6 排序

```python
a = np.array([2,1,3],dtype = 'float')

print(np.sort(a)) # 默认按从小到大排列 [1. 2. 3.]
```

输出排序好的数组当中的元素在原始数组当中的索引形成的数组：

```python
a = np.array([2,1,3],dtype = 'float')

print(np.argsort(a)) # 默认按从小到大排列 [1 0 2]
```

这里这么理解：新排序的数组是[1 2 3]，原数组是[2 1 3]，因此1在原来的数组当中的索引是1； 2在原来数组当中的索引是0；3在原来数组当中的索引是2，因此是1 0 2

------



##### 7.7 数组拼接

以两个数组的拼接为例：

```python
a = np.array([[1,2,3],[4,5,6]],dtype = 'float')

b = np.array([[7,8,9],[10,11,12]],dtype = 'float')

c = np.concatenate((a,b))

# c = [[ 1.  2.  3.] [ 4.  5.  6.] [ 7.  8.  9.] [10. 11. 12.]]
```

拼接时，参与拼接的多个元素需要用一个圆括号包裹起来：(a,b,c,...)，包裹后传入函数当中。

拼接时，要注意两个参与拼接的数组的某一维度的尺寸要相等，并且**拼接是可以选择维度的：axis值**，默认是0，即沿着纵向y轴方向拼接，此时要求列数相等。

------



##### 7.8 where函数

这个函数，传入的是一个含有numpy数组的逻辑表达式，**返回的是一个新的数组在原数组的索引**，下面举个例子：

```python
a = np.array([1,2,3],dtype = 'float')

print(np.where(a > 1)) # [1,2]
```

此时，表示返回一个新的数组，这个数组的元素满足：从a当中取出所有大于1的元素的**索引**

因此如果想要获取这些元素，只需要把**这个索引的数组传入原数组当中**，作为**花式索引**即可：

```python
a = np.array([1,2,3],dtype = 'float')

print(a[np.where(a > 1)]) # a[[1,2]] => [2.0,3.0]
```



##### 7.9 未完待续...



### Pandas

#### 1.Series

首先创建一个Series:

```python
import pandas as pd

s = pd.Series([1,2,3,4,5])
```

Series在pandas里类似于np.array或python中的list，即是一个类似列表的结构，只不过与列表最大的不同在于，Series的索引是显式给出的，下面是上方创建的s被打印的结果：

0    1
1    2
2    3
3    4
4    5
dtype: int64

左侧的一列是索引，这个索引是显式给出的，而且可以不连续、被修改为其他索引形式(A,B,C这样)，不过这个部分不是重点。

对Series的访问，和列表等很相似，都是通过[索引]进行访问，但是在访问之前，要确保索引是正确(指是数字，不是字母)而且连续的。

**Series也可以被转为list**，但是就目前看，意义并不大，因为对Series同样可以按索引访问到其中的元素。

------



#### 2.DataFrame的创建

DataFrame就是Series的容器，它是一个二维的表，其中每一列都是一个Series，**并且通常每一个Series都有一个表头的名称**，来表示这一列数据的含义。它的创建方式比较多，下面的代码可以创建一个简单的DataFrame:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(6,4))
```

​				0				1				2				3

0  0.955723  0.065832  0.051312  0.562031
1  0.446023  0.311044  0.498848  0.736114
2  0.079766  0.260437  0.969308  0.320395
3  0.276903  0.988961  0.606316  0.775523
4  0.159564  0.208592  0.101513  0.936167
5  0.990481  0.960844  0.320571  0.723348

行索引依旧是普通的数字，此时的列索引可以被重新命名为代表具体含义的名称。

另一种更常用的方式是通过字典创建DataFrame，这个字典其实是一种类字典的结构：

```python
import pandas as pd
import numpy as np

name_list = ['张三','李四','王五']

age_list = [18,19,22]

sex_list = ['男','女','男']

df = pd.DataFrame(
    {
        '姓名': name_list,
        '年龄': age_list,
        '性别': sex_list
    }
)
```

即通过一个大括号{}，传入不同的列，每一列使用 'key' : value_list的形式，这里要注意，每一列的list必须等长，否则会报错。

上面的代码可以生成这样一个DataFrame:

   姓名  年龄 性别
0  张三  18  男
1  李四  19  女
2  王五  22  男

------



#### 3.DataFrame的行操作

接下来是DataFrame的一些简单的行操作

##### 3.1查看头尾五行数据

使用下面的两行代码，可以分别查看DataFrame的最前面五行和最后五行：

```python
import pandas as pd
import numpy as np

print(df.head())
print(df.tail())
```

------



##### 3.2删除某行

删除某一行，语法是这样的：

```python
import pandas as pd
import numpy as np

df.drop(index,inplace=True)
```

其中，**inplace表示直接在原地对dataframe进行修改**(这个inplace在大部分pandas对dataframe操作的函数当中都有，建议都采用True的形式，节省代码)，而不只是返回一个修改后的对象，这样就不需要再额外将修改后的结果赋值了。

------



##### 3.3添加一行

通过下面的步骤，可以实现向一个dataframe当中添加一行的操作：

```python
import pandas as pd
import numpy as np

name_list = ['张三','李四','王五']

age_list = [18,19,22]

sex_list = ['男','女','男']

df = pd.DataFrame(
    {
        '姓名': name_list,
        '年龄': age_list,
        '性别': sex_list
    }
)

# 首先，创建一个字典，字典要包含该dataframe的每一列的属性：

dit = {'姓名':'李六','年龄':19,'性别':'男'}

# 接下来，将这个字典转为Series的类型

dit = pandas.Series(dit)

# 最后，使用_append()函数插入这一行

df = df._append(dit,ignore_index = True)

```

注意，最后的_append函数，要添加 **ignore_index = True** ,否则会报错。

------



##### 3.4行索引恢复

执行删除，或者一些分数据集的随机化行操作后，行索引会被打乱，此时执行下面的代码可以恢复行索引为顺序数列:

```python
import pandas as pd
import numpy as np

df.res_index(drop=True,inplace=True)
```

这里的drop设置为True，则旧的索引会被删除，如果不设置，则旧的索引会成为dataframe的新的一列，完全没有必要，因此可以直接drop掉。

------



#### 4.DataFrame的列操作

接下来是DataFrame的一些简单的列操作

##### 4.1删除某列

删除某一列的语法和删除某行的很相似，只是此时需要指明删除的方向：

```python
import pandas as pd
import numpy as np

df.drop('列索引名称',inplace=True,axis = 1)
```

这是因为axis默认是0，即删除的是行，需要显式设置为1方可执行列删除的操作。

------



##### 4.2添加一列

添加一列在dataframe中非常简单，完全类似于字典添加新元素，只需要**创建一个list，这个list要和dataframe中已有的列长度相同**，而后用下面的语法添加：

```python
import pandas as pd
import numpy as np

new_list = [xxxx]

df['new_list'] = new_list
```

------



#### 5.DataFrame的高级查找

##### 5.1单一条件查找

```python
import pandas as pd
import numpy as np

name_list = ['张三','李四','王五','周六','余七','刘八']

age_list = [18,19,22,10,14,12]

sex_list = ['男','女','男','男','男','女']

df = pd.DataFrame(
    {
        '姓名': name_list,
        '年龄': age_list,
        '性别': sex_list
    }
)
```

假设我们有上面这个dataframe，它打印出来是这样的：

   姓名  年龄 性别
0  张三  18  男
1  李四  19  女
2  王五  22  男
3  周六  10  男
4  余七  14  男
5  刘八  12  女

现在我们要通过高级查找的方式，找出所有的未成年人(年龄小于18)，这样的操作，过去可能通过循环，但是这里可以直接通过dataframe的高级查找，它的语法如下：

```python
df[df['年龄'] < 18]
```

即**df[查找条件]**，即可返回对应的行，**查找条件往往由某一列的值的一个布尔表达式充当(如本例)**，当执行完，即可拿到结果：

   姓名  年龄 性别
3  周六  10  男
4  余七  14  男
5  刘八  12  女

当然，查找的结果返回的是一个新的dataframe，如果我们想要这个新的，可以把它赋值给一个dataframe。(这样等同于**一次性删除掉了很多不想要的行**，因为只保留了剩下的行)，因此高级搜索有时可以用来删除大量不符合要求的行。

------



##### 5.2多条件查找

当同时想进行多个条件的联合查找时，可以用括号和逻辑符号对多个条件进行连接，语法如下:

```python
df[(条件1) 逻辑连接 (条件2) ...]
```

下面是一个实例

```python
import pandas as pd
import numpy as np

name_list = ['张三','李四','王五','周六','余七','刘八']

age_list = [18,19,22,10,14,12]

sex_list = ['男','女','男','男','男','女']

df = pd.DataFrame(
    {
        '姓名': name_list,
        '年龄': age_list,
        '性别': sex_list
    }
)

df[(df['年龄'] < 18) & (df['性别'] == '女')]
```

执行后可以搜索到结果：  

 	姓名  年龄 性别
5  刘八  12  女

------



##### 5.3两个DataFrame联合查找

有时候，我们有这样的需求：

有两个dataframe，它们有一列相同的属性列：id，现在我想看看第二个dataframe当中所有的id等于第一个dataframe 的id的那些行，此时就需要执行两个dataframe的联合查找，它的语法是isin：

```python
import pandas as pd
import numpy as np

# 假设我有一个df1和一个df2
df = df2[df2['id'].isin(df1['id'])]
```

此时就完成了两个dataframe的联合查找，这个可能用的不多，因此不需要过多理解。

另外，因为当前的需求是找出dataframe2当中的行，因此是df2[联合查找条件]，并且isin函数的调用者必须是df2，不能是df1

------



#### 6.DataFrame之Apply函数



#### 7.DataFrame缺失值处理

##### 7.1缺失值搜索

首先，我们可以用下面的代码，找出所有的缺失值所在的行：

```python
df[df['名字'].isnull()]
```

其实还是用到了前面提到的高级搜索，只不过搜索的条件用到了**.isnull()函数**，这个函数**会返回某个属性列所有的为空值的行的索引**信息，从而搭配高级搜索，可以搜到所有 名字 这一列为空值的行，这里以名字列为例，可以修改为任何其他属性列。

------



##### 7.2缺失值填充

对缺失值的处理，第一个方向是对其进行填充，填充可以使用下面的函数：

```python
df['年龄'].fillna(np.mean(df['年龄']),inplace=True)
```

fillna函数，会对缺失值进行填充，我们需要首先确定对哪一列进行填充，通过**列.fillna()调用函数**，而后**第一个参数是填充的值**，我们可以在这里**灵活的发挥**，例如用均值、中位数等等，第二个是inplace，这个不多解释。

------



##### 7.3缺失值删除

第二个方向就是直接删除掉缺失值，这里分为对缺失值所在的行或列进行删除：

```python
df.dropna(axis=0,inplace=True)
```

注意，这里的删除是对行进行删除，并且是所有属性列当中，每一个为空值的行都会被删除掉。

------



##### 7.4异常值处理搭配Apply函数



#### 8.DataFrame排序

pandas提供了对dataframe进行排序的函数，可以让整个dataframe按照某一列或某几列属性列的值进行排序，语法如下:

```python
df.sort_values(by=['评分','投票人数'],ascending=False,inplace=True)
```

注意，在方括号[]当中可以放置多个列索引，**并且会按照放置的先后顺序，决定以哪一个索引进行第一排序**，以此类推，ascending决定升序还是降序。

------



#### 9.DataFrame的合并

合并dataframe，可以使用concat函数，既可以执行按列合并，也可以按行合并，下面是一个按行合并的例子：

```python
import pandas as pd
import numpy as np

name_list = ['张三','李四','王五','周六','余七','刘八']

age_list = [18,19,22,10,14,12]

sex_list = ['男','女','男','男','男','女']

df1 = pd.DataFrame(
    {
        '姓名': name_list[:2],
        '年龄': age_list[:2],
        '性别': sex_list[:2]
    }
)

df2 = pd.DataFrame(
    {
        '姓名': name_list[2:],
        '年龄': age_list[2:],
        '性别': sex_list[2:]
    }
)

df = pd.concat([df1,df2],ignore_index=True)
```

df1长这样：

   姓名  年龄 性别
0  张三  18  男
1  李四  19  女

df2长这样：

   姓名  年龄 性别
0  王五  22  男
1  周六  10  男
2  余七  14  男
3  刘八  12  女

执行按行的合并后的df：

   姓名  年龄 性别
0  张三  18  男
1  李四  19  女
2  王五  22  男
3  周六  10  男
4  余七  14  男
5  刘八  12  女

注意，ignore_index是为了不保留旧的index，否则行索引会不连续，还需要再次重置，不如直接ignore旧的index，重新生成连续的index。

------



#### 终：DataFrame与pickle特别篇

##### 加载pickle

执行下面的代码，加载pickle文件为dataframe：

```python
df = pd.read_pickle(r'path')
```

##### 保存pickle

```python
df.to_pickle(r'path')
```

------



### Matplot



## 其他相关内容

### 1.python代码运行性能分析

首先，安装第三方包：snakeviz

```bash
pip install snakeviz
```

之后，在终端输入下面指令：

```bash
python -m cProfile -o target.prof your_code.py(此处要修改为待分析的代码文件名)
```

之后，对应的py文件会开始运行，运行一段时间后，即可得到一个target.prof文件，而后在终端输入这行指令看到可视化结果：

```bash
snakeviz target.prof
```

------



### 2.python进度条小工具

tqdm可以在控制台打印一个进度条，用来检测当前的运行进度，并实时返回预计运行剩余时间，需要两步使用该工具：

首先，导入tqdm:

```python
from tqdm import tqdm
```

而后，在**for循环的in关键字**后面套上tqdm即可：

```python
for i in tqdm(range(100))
```

另外，tqdm还可以传入参数：desc = ''，传入后，进度条前面会加上该desc的内容:

```python
for i in tqdm(range(100),desc = 'Description of the tqdm')
```

------



### 3.数据填充(Padding)

#### 3.1 观察数据分布

下面的代码可以辅助观察数据的分布:

```python
import matplotlib.pyplot as plt

plt.rc("font", family='Microsoft YaHei')

plt.hist(np.array(xxx), bins=30, color='blue', alpha=0.7)
plt.title('训练数据分布直方图')
plt.xlabel('数值范围')
plt.ylabel('频率')
plt.grid(True)
plt.show()
```

------



#### 3.2 填充与掩码生成

下面是一个通用的填充并生成掩码的代码，其默认所有的数据均是**一维**的、**tensor**类型，如不是，需要进行转换：

```python
max_length = 100 # 这里需要通过变量找到填充的最大长度

mask = torch.ones(max_length,dtype = bool)

if len(data) < max_length:

	pad_seq = torch.zeros(max_length - len(data))

	data = torch.cat((data,pad_seq),dim = 0)
    
    mask = (data != 0)
```



### 4.通过.sh脚本文件串行多任务炼丹

```bash
# 任务1
python ../src/main.py 运行参数1 运行参数2 ...

# 任务2
python ../src/main.py 运行参数1 运行参数2 ...

...
```

最后在终端，cd到脚本所在位置，运行下面指令开始执行多任务：

```bash
bash xxx.sh
```

