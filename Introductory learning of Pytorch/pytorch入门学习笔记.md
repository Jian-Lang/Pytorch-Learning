# **Pytorch入门学习笔记**

```html
<p>
    One day, you will do things that others can not do!
</p>
```

## **第一章：初识Pytorch**

### 预备式：安装Pytorch

进入pytorch的官网：https://pytorch.org/docs/stable/index.html

而后找到download部分，选择自己的系统，推荐选择pip的安装模式，之后到对应的anaconda环境下，利用pip指令进行安装。

### 1. 构建数据集

首先，要导入数据集的相关Python包：

```python 
from torch.utils.data import Dataset
```

接下来需要建立一个数据集类，继承Dataset父类

```python 
class MyDataSet(Dataset):
```

这类会包含(重写)以下几个部分：

#### **(1)** **初始化**(**构造函数)函数**

给这个数据类进行初始化，例如初始化数据的存储路径等信息。

示例代码:

```python 
def __init__(self,file_root):
    self.file_root = file_root
```

注：补充几个关于获取文件路径的函数

```python 
import os

# 列出path对应的文件夹下所有文件的文件名(注意给出的是文件名，不是文件路径)

path = "xxx"

dir_list = os.listdir(path)

# 拼接几个路径,下面的代码表示把path 1,2 3按照顺序拼接起来

path = os.path.join(path1,path2,path3)
```

#### **(2) 读取数据函数**

这个函数负责从指定位置把数据给读进来，并返回数据和标签(监督学习)

示例代码:

```python 
def __getitem__(self,index):
    img = Image.open(self.file_root)
    label = open("label.txt","r",encoding = "UTF-8").read()
    return img,label
```

#### **(3) 获取数据总量**

示例代码:

```python 
def __len__(self):
    return len(self.dir_list)
```

#### **(4) 数据重构**

将数据重新分成一个一个的块(batch)，这部分后续会有专门的一节来讲解

#### (5) **完整的构建数据集代码:**

示例代码:

```python 
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        image_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,image_name)
        img = Image.open(img_item_path)
        label = image_name
        return img, label

    def __len__(self):
        return len(self.img_path)
```

### 2. tensorboard

tensorboard是用来观察模型训练过程中损失函数变化趋势的一个工具

#### (1) tensorboard的使用方法

**首先，安装tensorboard：**

```bash
pip install tensorboard -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple
```

**接下来，导入tensorboard：**

```python
from torch.utils.tensorboard import SummaryWriter
```

**初始化一个summarywriter对象**，用于执行数据、图像的写入操作，以便生成图像：

```python
writer = SummaryWriter("logs")
```

上面初始化代码中，传入参数"logs"代表要**生成一个叫logs的文件夹**，这个文件夹里面会存放后续的**事件文件**(tensorboard事件文件)

也可以表示存在一个logs文件夹，后续的事件文件都放在logs文件夹下

之后我们可以简单的**编写一个y = x的图像**：

```python
for i in range(100):
    writer.add_scalar("y = x",i,i)
```

其中，**add_scalar()函数**是用来添加图像及图像中的数据的，第一个参数代表图像的title，第二个和第三个分别代表x和y轴的数据

**最后别忘了给writer关闭一下**：

```python
writer.close()
```

于是上面部分的完整代码是：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y = x",i,i)
    
writer.close()
```

运行这段代码，我们会发现多了一个logs文件夹，并且这个文件夹下有events文件，这就是用tensorboard生成的文件

#### (2) 查看tensorboard生成的事件

**接下来是查看生成的事件文件的方法：**

首先**打开anaconda的命令框**，之后输入下面的指令

```python
tensorboard --logdir=logs文件夹的绝对路径 --port=8009
```

其中logdir要写**events文件的父文件夹的绝对路径**(这样最保险)，这里自然是logs文件夹的绝对路径

之后会生成这样的一段文字：

> TensorFlow installation not found - running with reduced feature set.
> Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
> TensorBoard 2.13.0 at http://localhost:8009/ (Press CTRL+C to quit)

点击链接即可访问，如果想要终止访问，按CTRL + C

#### (3) tensorboard添加图像

**接下来，补充一下tensorboard如何添加图像(上面的例子是添加数据)：**

tensorboard添加并打开图像，需要用到函数add_image(),这个函数有几个重要参数：图像名称，图像，步长

第一个和第三个参数都比较容易填写，**第二个参数图像，需要使用几种特定的参数**，**numpy和tensor**是常见的支持类型。我们这里采用**numpy格式**的图像参数

**下面这段代码演示了如何把一个图像参数转成numpy格式的图像参数**

```python
from PIL import Image
import numpy as np

image_path = "xxx"

img = Image.open(image_path)

img_array = np.array(img)
```

其中比较关键的函数是np.array,能够把原始图像参数转成numpy格式的图像参数

于是我们就可以正常调用add_image函数来为tensorboard添加图像了，下面是完整的代码：

```python
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

image_path = "xxx"

img = Image.open(image_path)

img_array = np.array(img)

writer.add_image("image_title",img_array,2,dataformats='HWC')

writer.close()
```

注意，上面的add_image使用了第四个参数：dataformats，这是因为在add_image方法中，dataformats参数的默认值是CWH，即先通道，然后是宽度，最后是高度，但是我们转换得到的numpy类型的图像，它的format是先高度，再宽度，最后才是通道，所以需要把这个参数的值进行重新设置(覆盖默认值)

### 3.transforms

#### (1) transforms的简介

首先，什么是transforms呢，transforms其实是对图像数据集进行处理的一个工具箱，**注意，它是一个工具箱，不是一个工具**，这是有区别的，因为它里面还有很多的工具，或者更准确的说，**它里面有很多工具类**，我们用它里面工具类对图像数据进行各种各样的处理。

![image-20230619165043276](C:\Users\Lobster\AppData\Roaming\Typora\typora-user-images\image-20230619165043276.png)

#### (2) 图像数据的三种类型及转换方法

在机器学习中，图像数据通常以三种形式出现，分别是**PIL、Numpy和Tensor**，其中Tensor是最广泛使用的，因为它是用矩阵来表示图像的形式。这里首先介绍PIL和Numpy形式的图像如何得到：

**PIL格式图像的获取:**

```python
from PIL import Image

image_path = "xxx"

img = Image.open(image_path)
```

此时这个img就是PIL格式的图像了，于是PIL库是用来把图像转成PIL 格式的工具库。

**Numpy格式图像的获取:**

```python
import cv2

img_path = "xxx"

img = cv2.imread(img_path)
```

此时这个img是Numpy格式的图像，即OpenCV库可以用来获取Numpy格式的图像。(前面上一章用Numpy库也可以，但没有这个方便)

#### (3) transforms工具箱的使用

这部分的重头戏来了：如何使用transforms这个工具箱呢？首先，我们要导入工具箱：

```python
from torchvision import transforms
```

其实从导入就能看出一些门道：我们不是从torch库中导入transforms，而是从torchvision中导入，那么有什么区别呢？torchvision其实是torch在视觉方面的分部，而我们transform刚好是处理图像的，所以它属于视觉分部中的子工具箱，因此我们是从torchvision中导入transforms。

接下来，我们介绍几种transforms工具箱中的工具类:

##### (3.1) ToTensor工具类

ToTensor工具类是用来把图像转成tensor格式数据的工具，也刚好把上面留下的最后一种图像格式填补了。下面是它的示例代码:

```python
from torchvision import transforms

trans_totensor = transforms.ToTensor()

img_tensor = trans_totensor(img) # 这个img是PIL或Numpy格式的图像对象
```

其中，第一行导入了transforms，第二行，我们创建了ToTensor类的对象(注意，根据前面Python基础所学内容，我们导入的是transforms，因此实例化它里面的类，需要用transforms.xxx)

最后，我们传入img对象，输出该img对象的tensor格式数据。**这个img对象既可以是numpy类，也可以是PIL类！**

##### (3.2) Normalize工具类

Normalize工具类是对图像进行归一化处理的工具，下面是示例代码:

```python
from torchvision import transforms

trans_normalize = transforms.Normalize([1,2,3],[4,5,6])

img_norm = trans_normalize(img)

https://blog.csdn.net/Yasin1/article/details/120123710
```

我们注意到，在初始化Normalize类的对象时，我们传入了两个列表[1,2,3]和[4,5,6]，这是由于它的构造方法要求必须传入两个参数，分别代表**归一化时图像每个通道上的均值和标准差**，即**[1,2,3]代表三个通道的均值分别是1,2,3，[4.5.6]代表三个通道标准差是4,5,6**

**这里传进去做归一化的图像对象必须是tensor类型的对象！！！**不能是PIL或者Numpy，输出的对象也是tensor类型的对象

##### (3.3) Resize工具类

Resize工具类是对图像的尺寸进行重新设置的工具，下面是示例代码:

```python
from torchvision import transforms

trans_resize = transforms.Resize((100,100))

img_resize = trans_resize(img)
```

在初始化Resize工具类的对象时，我们同样需要传入参数，**(100,100)代表处理后的图像的尺寸是100 x 100**

我们也可以只传入一个参数，例如我们只传入一个100，则它会把图像的短边按100来缩放，另一边等比缩放。

**传入的img对象可以是PIL或tensor类型的。**

##### (3.4) RandomCrop工具类

RandomCrop工具类是对图像进行随机裁剪的工具，下面是实例代码:

```python
from torchvision import transforms

trans_crop = transforms.RandomCrop((100,100))

img_crop = trans_crop(img)
```

在初始化RandomCrop工具类的对象时，我们同样需要传入参数，**(100,100)代表我们要裁剪的碎片的尺寸100 x 100**

**传入的img对象可以是PIL或tensor类型的。**

##### (3.5) Compose工具类

从上面四个工具类不难看出，我们可能会对图像进行多个操作，例如我们有可能先把图像转成tensor类型，之后再对图像进行归一化、缩放等，此时如果新建多个工具类，会增加额外代码量，为了简化代码量，Compose工具类就出现了，它能够把多个工具类集成到自己的类里面，以一敌百！下面是它的示例代码:

```python
from torchvision import transforms

trans_compose = transforms.Compose([
    transforms.ToTensor(), transforms.Resize(100,100)
])

img_trans = trans_compose(img)
```

要注意的是，我们**输入的img对象**，要**满足Compose中第一个工具类的输入**； **输出的结果**则**对应Compose最后一个工具类**的输出

因此我们可以这样理解Compose工具类的处理流程：

![image-20230619165007057](C:\Users\Lobster\AppData\Roaming\Typora\typora-user-images\image-20230619165007057.png)

#### (4) 号外：__getitem__、__call__和forward函数

##### (4.1) __getitem__函数

getitem函数是**类的魔术方法**，它是用来返回参数的，但是它特殊的地方在于，**当我们设置函数的传参时，这个传参可以通过另一种方式传入函数体**，语法是：**对象名[参数]**

举一个例子:

```python
class Car:
    def __getitem__(self,index):
        test_list = [1,2,3,4]
     	return test_list[index]
    
car = Car()

print(car[0]) # 对象名是car，所以我们用 car[0] 等价于调用car.__getitem__(0)
```

**极大的减小了代码量！**

##### (4.2) call函数

call函数是另一个类的魔术方法，它的特点是被调用时，可以用类名(参数1,参数2)的形式直接调用它，举一个例子:

```python
class Car:
    def __call__(self,p):
        print(p)
    
car = Car()

car("call函数演示") # 等价于调用car.__call__("call函数演示")
```

极大的减小了代码量！上面四种工具类几乎全部是用call函数或下面的forward函数实现的功能，但是都可以直接类名(参数调用)

这种方法可以视为把类的成员函数当做了一个普通的函数来调用(普通函数只需要函数名()调用，这里是类名()调用，很类似)

##### (4.3) forward函数

forward函数完全等同于call函数！但是注意，我们只能在继承了nn.Model的类中重写forward()

#### (5) 尾声：如何自学transforms的其他操作

- 通过ctrl键并点击transforms，可以看到它里面的各种工具类，我们主要关注以下几个点：
- 这个类是做什么的
- 这个类初始化需要哪些参数(有默认值的参数不需要管，只看没有默认值的参数)
- 这个类在操作图像时，输入是什么类型 (tensor，PIL还是numpy ),输出是什么类型？

## 4. pytorch提供的图像数据集及使用方法

在本章第一节，我们介绍了如何把自己的图像数据做成数据集来提供给模型，这一节我们学习pytorch提供的图像数据集如何使用，这些数据集不需要我们去构建，我们可以拿来练手或做他用。

#### (1) 下载并导入数据集

首先，pytorch提供的图像类数据集可以在这里查到：

[数据集]: https://pytorch.org/vision/0.8/datasets.html	"pytorch数据集网站"

我们通过网站，可以看到各种图像数据集，这里以数据集CIFAR为例，图像是各种动物，因此是用来训练动物图片分类的模型的数据。

于是我们先导入图像数据集相关的模块：

```python
import torchvision
```

这里我们之前介绍过，凡是和图像有关的，基本上都在torchvision中。

接下来，我们来获取数据集:

```python
train_set = torchvision.datasets.CIFAR10(root = "数据集存放路径",train = True, download = True)

test_set = torchvision.datasets.CIFAR10(root = "数据集存放路径",train = False, download = True)
```

可以看到数据集的获取语法是通过**torchvision下的datasets下的具体数据集类**的实例化实现，在实例化时，需要传入一些参数：

**root参数代表数据集要放在哪里；train参数用于判断当前的数据集是训练还是测试的，True为训练，False为测试，最后download则表示是否要下载数据集。**

当我们第一次使用某个数据集时，download需要设置为True，来下载数据集，而下载时会提供一个链接，我们可以通过链接自行下载后，把数据集压缩包放在root对应路径下，而后再次运行即可(通过Python下载则太慢了), **第一次设置True运行成功后，以后可以不设置download参数**

#### (2) 数据集基本操作介绍

数据集获取之后，**它可以看做是一个大的列表**，其中**每一个列表元素又是一个二元组的列表**，第一个元组是图像数据本身，第二个是它的标签：[元素1[图像1, 标签1], 元素2[图像2，标签2],...]

例如我们可以通过这种方式获取测试集的第一个图片数据和它对应的标签：

```python
img,label = train_set[0]
```

这种方式获取到的图像数据img是PIL类型的

细心的朋友可以看到，这里获取图像和它的标签的方式，与第一节的案例是完全相同的，即我们在第一节的确自定义了一个数据集，和pytorch定义的，操作起来一模一样！

#### (3) 图像类数据集搭配transforms实现预处理

最后是将上一节的内容和这一节做一个拼接，即如何结合transforms和这节课的pytorch提供的图像数据集。

首先，在加载数据集时，实例化对象的过程中，除了刚才提到的几个参数，还有一个参数：**transform = xxx**

```python
train_set = torchvision.datasets.CIFAR10(root = "存放路径",train = True, transform = xxx, download = True)

test_set = torchvision.datasets.CIFAR10(root = "存放路径",train = False, transform = xxx, download = True)
```

这个transform参数，让我们传的正是transforms工具箱中的任意一个工具类，传入之后，整个儿数据集都会被这个工具类所处理。

于是有两种选择：

**直接传入某一个工具类：**

```python
train_set = torchvision.datasets.CIFAR10(root = "存放路径",train = True, transform = transforms.ToTensor(), download = True)
```

**通过Compose联合多个工具类：**

```python
compose_trans = transforms.Compose([xxx,xxx,xxx])

train_set = torchvision.datasets.CIFAR10(root = "存放路径",train = True, transform = compose_trans, download = True)
```

上节课讲到的这个transforms，经常被用在这个情境中，因此必须把这两节的内容串联起来理解！

## 5.DataLoader()介绍

在之前的章节中，我们总是认为数据是一个一个被加载进模型中的，但是在听了李宏毅老师的课之后，我们知道数据是要分批次的，也就是分**batch**的。

每一个batch中，有若干笔数据，当所有的batch都做过一遍之后，我们称为一个**epoch**，下图是数据、batch和epoch的关系：

![image-20230711161824048](C:\Users\Lobster\AppData\Roaming\Typora\typora-user-images\image-20230711161824048.png)

这里我们假设一共有六个数据，每两个数据分一个batch，因此**epoch = 3 batch**，即一个epoch中有三个batch，每一个batch做一次参数的迭代，因此**一个epoch后参数被更新了三次**，我们在训练中，会做**多个epoch**来得到最终的模型！

上面分batch的过程，在pytorch中可以用**Dataloader()类**来实现，以下是创建dataloader对象的示例代码:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset = test_set,batch_size = 2,shuffle = True,num_workers = 0,drop_last = False)
```

第一个参数**dataset**，表示要分batch的是哪个数据集；

第二个参数**batch_size**，是一个batch包含几个数据，这里设置2，即一个batch有两个数据

第三个参数**shuffle**，指的是每个epoch中是否打乱batch的分组。例如在第一个epoch中，数据1和2被分为1组，在下一个epoch中，是否要和第一个epoch的分组情况不同，True代表打乱分组，False则代表每次分组情况都相同。

第四个参数num_workers设为0即可

第五个参数**drop_last**表示当数据集不能整除batch_size时，最后剩余的数据是否要保留，False表示保留，True表示丢弃最后剩余不足一组的数据。

这个**dataloader对象内部分好了一个一个batch的对象**。这个类实例化后的对象，是这样的结构：

![image-20230711162415035](C:\Users\Lobster\AppData\Roaming\Typora\typora-user-images\image-20230711162415035.png)

**Dataloader对象并不是一个列表结构，即我们并不能通过dataloader[index]来访问它里面每一个batch的数据**，而是通过for循环：

```Python
for data in dataloader:
    imgs,targets = data
```

注意，imgs和targets，可以视为列表，它们的长度 = batch_size，在上面的例子里，它们的长度 = 2，即我们可以通过imgs[0]访问到每一个batch中第一个图像数据。

在训练中，我们往往是把每个batch的数据一起丢给模型，即每个imgs和targets丢给模型，算一次参数更新。

## 第二章：卷积神经网络(CNN)

### 1. 搭建自己的模型

首先，进入这一章，就正式开始搭建神经网络模型了。首先，我们先了解一下如何搭建一个网络模型:

第一步，我们需要先导入工具包:

```Python
from torch import nn
```

第二步，搭建模型，所谓的模型其实就是一个类，只不过需要继承神经网络模型父类：

```Python
class MyModule(nn.Module):
    # 初始化模型参数
    def __init__(self):
        super().__init__()
    
    # 预测结果，即模型最核心的部分
    def forward(self, x):
        return x + 1
```

到这里，我们的一个简单的模型就搭建好了，我们接下来尝试去用一下：

```Python
mymodule = MyModule()
x = torch.tensor(1)
print(mymodule(x))
```

输出是一个张量的2，表示运行顺利。

从搭建过程可以看出，一个神经网络模型，基本上要具备两个大的部分：

第一个部分是**初始化部分**，这部分主要对模型的一些参数进行初始化，当然第一步是先把父类初始化一下，然后把训练好的参数放进去

第二个部分就是计算**预测结果**，也就是我们要对input计算一个output，这是整个模型最重要的一步。

### 2. 二维卷积运算

首先，第一步自然的引入相关的包：

```Python
import torch.nn.functional as F
```

这里补充一句，这一章开始，我们所有关于**神经网络模型的包**，大部分都在**torch下的nn包**里面。这次导入的functional包，是nn包下的一个子包，**这个functional包里面里面主要是关于神经网络的一些有用的函数**。

接下来，我们分别定义卷积核和输入图像(在真实的神经网络模型中，卷积核的参数是默认已知的，因为**它是通过训练后确定的**，我们这里给出的卷积核是一个样例)

```Python
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]
                      ])

kernel = torch.tensor([
    [1,2,1],
    [0,1,0],
    [2,1,0],
])
```

定义完后，我们理论上可以调用F里面计算二维卷积的函数了，但是由于这个函数需要传入的input图像和kernel的张量，是包含四个维度参数的，即：batch_size, channels和宽、高，但是我们通过torch.tensor得到的张量只有宽高两个参数，因此这里需要用torch包里的reshape函数来进行参数重定义：

```Python
import torch

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
```

**reshape函数**的四个参数，刚好是：**batch_size, channels和高、宽**

接下来，只需要**调用函数计算卷积**即可：

```Python
output = F.conv2d(input,kernel,stride = 1,padding = 0)
```

其中，stride和padding是可以自定义的。

**reshape函数和前面讲过的resize并不相同：**

**resize是把图像的长和宽的尺寸进行强制修改，而reshape是保证图像的四个参数的积固定的基础上，四个参数之间进行的参数值的内部迁移！！！**

### 3. 卷积层

接下来，开始按照CNN的组成，一层一层介绍。首先，第一层是卷积层(conv)，这一层的主要作用是做特征提取，每一种卷积核提取一种特征。

我们按照本章第一节的方法，首先搭建我们的网络框架:

```Python
from torch import nn

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self,input):
        pass
```

接下来，我们要在我们的神经网络中加入卷积层，怎么加呢？是这样的一个思路：

在pytorch中，为我们提供了一个计算二维卷积的类：Conv2d类，因此我们只需要:

(1) 在初始化时，将Conv2d类做一个初始化

(2) 在forward函数中，调用Conv2d类的对象进行卷积操作

于是我们只需要将上面的代码这样修改:

```Python
from torch import nn
from torch.nn import Conv2d

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 在类与对象章节，我们学到过，在Python中，__init__构造函数中可以直接定义类的成员变量，定义方法为self.xxx
        self.conv2d = Conv2d()
        
    def forward(self,input):
        output = self.conv2d(input)
        return output
```

看起来还是很简单的，但是上面的代码显然不全对，原因是我们的Conv2d类的对象初始化没有传入参数，既然是要做卷积运算，那传入的参数肯定和二维卷积运算有关:

**Conv2d(in_channels = xxx,out_channels = xxx,kernel_size = xxx,stride = xxx,padding = xxx)**

其中:

in_channels表示我们输入的图像(数据)的通道数

out_channels表示想要输出的通道数，显然代表了需要几种卷积核，例如out_channels = 3,代表三种卷积核参与卷积运算

kernel_size表示卷积核的尺寸，输入一个值，则是方形，输入一个tuple，则可以自定义宽高

最后两个参数我们很熟悉了，就不做赘述了。这里需要补充的一点是，卷积核参数并不需要我们提供，这是因为这些参数本来也是需要训练得到的，我们在后面的部分会讲解参数训练，这里的卷积核由Conv2d类用特殊的方式来生成，我们只需要提供一个size。

最后是完整的带卷积层的神经网络：

```Python
from torch import nn
from torch.nn import Conv2d

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 在类与对象章节，我们学到过，在Python中，__init__构造函数中可以直接定义类的成员变量，定义方法为self.xxx
        self.conv2d = Conv2d(in_channels = 3,out_channels = 6,kernel_size = 3,stride = 1,padding = 0)
        
    def forward(self,input):
        output = self.conv2d(input)
        return output
```

我们可以用之前学过的Python的数据集来试一下这个网络的输出：

```Python
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision

dataset = datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset = dataset,batch_size = 64)

mymodule = MyModule()

for data in dataloader:
    imgs,targets = data
    output = mymodule(imgs)
```

如果想让输出可视化，以便于我们的观看，可以加入之前学过的tensorboard：

```Python
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch

dataset = datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset = dataset,batch_size = 64)

mymodule = MyModule()

writer = SummaryWriter("logs")

step = 0

for data in dataloader:
    imgs,targets = data
    output = mymodule(imgs)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("title",output,step)
    step += 1
    
writer.close()
```

这里有一个点，我们单独讲一下，即为什么要进行reshape操作：

tensorboard在展示图像时，只能展示三通道图像，即RGB图像，但是我们之前设置的输出通道数 = 6，因此我们需要进行一个reshape操作，在reshape时，通道数，宽和高我们都可以自主设定，但是第一个参数batch_size，我们如果不想计算时，可以设置为-1，此时Python会帮我们计算batch_size，计算方法：**四个参数的积 = output的参数总数量**

### 4. 池化层

池化层(Pooling)，是紧接着CNN的卷积层之后的层次，这个层最重要的作用是做数据的降维，以提高训练的效率，但是缺点也很明显，会损失一些数据。

池化层的编写套路和卷积层相同，我们还是先准备一个神经网络的框架：

```Python
from torch import nn

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self,input):
        pass
```

之后我们也猜到了，有一个专门做二维池化操作的类，我们在构造函数中初始化这个类，并在forward函数中调用这个类的池化操作即可。

这里补充一点：池化操作不像卷积操作一样，是一个固定方法的操作，而是有很多种池化操作，例如max pooling，mean pooling等等。我们这里举一个max pooling的例子，即把**每个区域的最大值当做这个区域的代表**的下采样操作。

```Python
from torch import nn
from torch.nn import MaxPool2d

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size = 3,ceil_mode = True,stride = 3)
        
    def forward(self,input):
        output = self.maxpool(input)
        return output
```

这里补充说明一下，这个MaxPool2d在初始化时，需要提供三个参数：

kernel_size是池化层的size，相当于要在多大的区域内池化

ceil_mode表示向上还是向下取整，True表示向上取整，即当右移后有部分空间是没有元素的，是否要采样，True表示采样，False则丢弃。

stride不做赘述，但是在pooling中，stride默认值取得是kernel_size的值，而非其他值

### 5. 非线性激活层

从池化层出来，要开始做预测操作了：非线性激活层(通过激活函数，来完成最终的分类)，这一层一般选用一些**非线性的激活函数**，我们以ReLu和Sigmoid函数为例来讲解一下：

```Python
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        
    def forward(self,input):
        output = self.relu(input)
        # output = self.sigmoid(input)
        return output
```

激活函数的初始化，比较简单，无需任何传参到构造函数中。

这里要补充一下：

**之前介绍的那些层，包括这个非线性激活层，它们在操作时，都有自己的默认tensor数据的shape，例如之前几个层要求tensor数据要有四个维度的shape，而这个激活层需要tensor有batch_size维度，在处理这类问题时，一定要确保自己传入的数据符合维度的要求。**

### 6. 线性层

在不断的卷积和池化后，激活函数之前，我们需要先做线性处理：**输入一维的数据，输出一维的结果**，中间涉及到维度的变换。

举一个例子，例如我们想要做的是图像识别，而我们一共有十种样品，那么在线性层，我们会接收来自上一层的输入，例如是1维的数据，有100个，然后输出是一维的数据，有10个。

下面是一个线性层的例子:

```Python
from torch import nn
from torch.nn import Linear

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear = Linear(100,10)
        
    def forward(self,input):
        output = self.linear(input)
        return output
```

其中，Linear()初始化时，需要传入两个参数，**分别代表输入的一维数据的宽度，和输出的一维数据的宽度**。

在我们实际的情境中，当经过了卷积/池化后的数据，不可能是一个一维的，而是多个维度的，那么我们可以用下面的两种方式来把数据拉平：

```Python
data = torch.flatten(data)

data = torch.reshape(data,(1,1,1,-1))
```

拉平后，我们通过shape看到data的宽度，就可以设置Linear()的初始化参数了。

### 7. Sequential的使用

Sequential作为本章的收尾，它很类似于前面第一章的compose，即它是用来打组合拳的。

我们学了卷积层、池化层、非线性激活层、线性层，而在真实的神经网络模型中，往往需要这些层搭配起来，因此Sequential就显得非常重要了。下面是Sequential的使用案例：

```Python
from torch import nn
from torch.nn import Linear,Conv2d,MaxPool2d,Sequential,Flatten
from torch

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = Sequential(
        	Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )
        
    def forward(self,input):
        output = self.model(input)
        return output
```

可以看到它的语法也是很简单的，只需要在初始化Sequential对象时，把所有需要的层的对象全部初始化在里面即可。如果不用Sequential，则需要一行一行定义多个对象，并且需要在forward中一行一行处理，代码量十分惊人。

其中Flatten类的对象，它做的事情和torch.flatten一样，只不过我们可以直接在模型内就把数据拉平，而不需要单独再外面在写一行。

### 8.损失函数

损失函数Loss Function是预计输出和实际输出之间差距的函数，常见的有0-1损失函数、平方差损失函数、交叉熵损失函数等，下面介绍如何在pytorch中使用Loss Function：

#### (1) L1 Loss Function

这种损失函数，公式比较简单：

F = 1/N(Σ|Xi - Yi|) **或** F = Σ|Xi - Yi|, **N代表batch_size **(损失函数默认是计算一个batch的值)

代码示例:

```Python
from torch.nn import L1Loss
import torch

outputs = torch.tensor([1,2,3],dtype = torch.float32)
targets = torch.tensor([1,2,4],dtype = torch.float32)

loss_func1 = L1Loss(reduction = 'mean')
loss_func2 = L1Loss(reduction = 'sum')

loss1 = loss_func1(outputs,targets)
loss2 = loss_func2(outputs,targets)

print(loss1) # 1 / 3
print(loss2) # 1
```

首先导入LILoss损失函数的类，然后初始化一个该损失函数的对象即可，其中我们可以初始化时加上参数reduction，选择计算平均值还是求和，对应上面的公式。

#### (2) MSE Function

平方差损失函数，它的定义式：

F = (Xi - Yi)²，如果是一个batch进行训练，F = 1/N ( Σ(Xi - Yi)² )

这个损失函数显然可以看做上面那个损失函数的进阶版：进阶为一个可导函数，从而便于我们后面的优化工作(梯度下降法)

代码示例:

```Python
import torch
from torch.nn import MSELoss

outputs = torch.tensor([1,2,3],dtype = torch.float32)
targets = torch.tensor([1,2,5],dtype = torch.float32)

loss_func = MSELoss()

loss = loss_func(outputs,targets)

print(loss) # (5 - 3)²  / 3
```

#### (3) CrossEntropy Function

交叉熵损失函数，它的定义式比较复杂:

![image-20230704174005206](C:\Users\Lobster\AppData\Roaming\Typora\typora-user-images\image-20230704174005206.png)

其中，x代表当前class对应位的概率，class相当于这里是标签；所以当预测准确时，x应该较大，从而x[class]比较大，loss整体比较小。

举一个例子，假设一个三分类问题：人，狗和猫

输入一个图片，标签class = 1，即这张图是狗

我们的预测值x = [0.1,0.5,0.4]，表示人的概率是0.1，狗0.5，猫0.4

那么这里x[class] = 0.5 x 1

代码示例：

```Python
import torch
from torch.nn import CrossEntropyLoss

outputs = torch.tensor([0.1,0.5,0.4])

targets = torch.tensor([1])

outputs = torch.reshape(outputs,(1,3))

loss_func = CrossEntropyLoss()

loss = loss_func(outputs,targets)

print(loss)
```

注意，crossentropy损失函数，需要限定预测值和实际值的参数的shape：

预测值，**需要有一个batch_size和一个种类数**，相当于要知道有几类，然后有几笔数据，因为预测值在每一类都有一个概率

真实值则**只需要一个batch_size**，因为真实值直接对应一个具体的类别，不需要知道有几类

### 9.优化方法(优化器)

常见的优化方法是梯度下降法，原理表达式：

θt = θt-1 - α * 1/N * Σ(dL/dθ)

表达式涉及到：

**α：学习率**，是一个超参数，我们需要手动调参

**dL/dθ：参数偏导数**，需要先选定合适的损失函数

**N: batch_size**，一般来说，N越大，训练速度越高，但是越容易得到一个尖锐的局部最小值，N越小，训练速度越低，越容易得到一个平缓的局部最小值。

下面展示随机梯度下降法(SGD)的代码：

```python
from torch.optim import SGD

module = Mymodule()

optim = SGD(mymodule.parameters(),lr = 0.01)
```

在初始化SGD对象时，一共需要两个参数，第一个是模型的参数，直接用模型.parameters()即可，第二个是学习率，lr一般设为0.01

一般而言，训练参数(优化参数)分为以下几个步骤：

定义损失函数

确定优化器

反向传播计算偏导数

更新参数

下面是完整的训练参数的代码示例：

```python
"""
@author: Lobster
@software: PyCharm
@file: nn_optimise.py
@time: 2023/7/5 16:34
"""

import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential,CrossEntropyLoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, input):
        output = self.model1(input)
        return output

if __name__ == '__main__':

    dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

    dataloader = DataLoader(dataset=dataset,batch_size=64)

    loss_cross = CrossEntropyLoss()

    mymodule = MyModule()

    optim = SGD(mymodule.parameters(),lr=0.01)

    for epoch in range(20):
        for data in dataloader:
            imgs, targets = data
            outputs = mymodule(imgs)
            loss = loss_cross(outputs, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
```

其中，关键的几步：

loss = loss_cross(outputs, targets) ：计算损失

optim.zero_grad()：梯度清零

loss.backward()：反向传播计算梯度

optim.step()：更新参数

### 10.使用并修改现有的CNN网络模型

pytorch提供了一些现有的模型，下面我们来尝试直接使用Pytorch的CNN网络模型: vgg16

```python
from torchvision.models import vgg16

model = vgg16(pretrained = False)

print(model)
```

上面代码导入并使用vgg16模型

其中，pretrained表示是否要进行预训练，如果要预训练，那么就会通过一些数据对模型的参数进行特殊的初始化，如果不预训练，则模型的参数就会保持初始值。

打印模型，我们发现vgg16模型是这样的:

```python
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

此时我们如果想要再添加一层，可以用下面的语句：

```python
from torchvision.models import vgg16

model = vgg16(pretrained = False)

# model.add_module('层名称',该层的对象)
model.add_module('linear',Linear(1000,10))

model.classifier.add_module('linear',Linear(1000,10))
```

上面有两种添加方法，第一种直接在模型最后加上我们新的层，第二种是在指定的Sequential中添加我们新的层，这个根据需要自行选择，第二种需要指定哪一个sequential。

我们也可以**修改某一层**：

```python
from torchvision.models import vgg16

model = vgg16(pretrained = False)

model.classfier[6] = Linear(1000,10)
```

这样就把model的classfier下第六层进行了修改。

## 第三章：模型与训练方法

### 1.保存和加载网络模型

保存和加载网络模型，一般而言有两种方法：

#### 第一种方法：保存完整的模型，加载完整的模型

```python
# 假设model是一个训练好的vgg16模型

torch.save(model,"model.pth")

new_model = torch.load("model.pth")
```

这样把整个模型放进save中，就是保存了整个模型，加载时直接加载整个模型即可。**注意，模型保存为pth后缀**即可。

#### 第二种方法：保存模型参数，加载模型参数

这种方法比起上一种，节约了一些空间，提高了效率：

```python
# 假设model是一个训练好的vgg16模型

torch.save(model.state_dict(),"model.pth")

new_model = vgg16(pretrained = False)

new_model.load_state_dict(torch.load("model.pth"))
```

第二种的思路是，保存模型的参数，然后当需要加载它时，需要先创建模型的大框架，即创建一个没有pretrain的空模型，然后把之前保存的参数加载进去。

### 2.训练模型的完整套路(重点***)

完整训练模型的套路大概是这样的：

(1) 加载数据

(2) 数据分批次

(3) 定义损失函数和优化器

(4) 导入待训练的模型

(5) 设定训练的参数

(6) 训练模型，并记录训练参数

(7) 可以做一个训练结果可视化

(8) 保存模型

下面代码是一个完整的模型训练示例代码:

```python
# 导入库文件

import torch
import torchvision
from torch.utils.data import DataLoader
from CIFAR_Model import MyModule
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

# 定义训练设备

device = torch.device("cuda")

# 导入数据集

train_data = torchvision.datasets.CIFAR10(root="dataset",train=True,transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="dataset",train=False,transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)

test_data_size = len(test_data)

# 数据集重构

train_data_loader = DataLoader(dataset=train_data,batch_size=64)

test_data_loader = DataLoader(dataset=test_data,batch_size=64)

# 导入待训练模型

untrained_model = MyModule()
untrained_model = untrained_model.to(device)

# 定义损失函数

loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 定义优化方法

learning_rate = 0.01

optim = SGD(params=untrained_model.parameters(),lr=learning_rate)

# 定义训练中的参数

total_train_step = 0

total_test_step = 0

accuracy = 0

# 训练过程可视化：加入tensorboard

writer = SummaryWriter("logs")

# 模型训练

epoch = 20

for i in range(epoch):
    print(f"-------------第{i + 1}轮训练开始-------------")

    # 训练部分
    untrained_model.train()
    for data in train_data_loader:

        # 提取input和label：
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)

        # 预测输出结果
        outputs = untrained_model(images)

        # 计算损失
        loss = loss_fn(outputs,labels)

        # 通过损失，优化参数
        optim.zero_grad()
        loss.backward()
        optim.step()

        # 训练次数计数
        total_train_step += 1

        # 可视化训练损失
        writer.add_scalar("训练损失",loss,total_train_step)

        if total_train_step % 100 == 0:
            print(f"第{total_train_step}轮训练后的损失loss = {loss}")

    # 测试部分
    untrained_model.eval()
    # 定义一些测试参数
    correct_count = 0
    total_test_loss = 0
    # 测试时，不进行梯度的计算和更新
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = untrained_model(images)
            # 计算正确预测的次数
            correct_count += ((outputs.argmax(1) == labels).sum())
            loss = loss_fn(outputs, labels)
            total_test_loss += loss

    accuracy = correct_count / test_data_size
    total_test_step += 1
    print(f"第{total_test_step}轮测试的总损失loss = {total_test_loss}")
    print(f"第{total_test_step}轮测试的正确率accuracy = {accuracy}")
    writer.add_scalar("测试损失", total_test_loss,total_test_step)
    writer.add_scalar("测试正确率", accuracy,total_test_step)
    
    torch.save(untrained_model, f"CIFAR10_model_x{i}.pth")
    print("模型已保存")
    
writer.close()
```

### 3.GPU加速训练

之前训练模型，我们使用的是CPU的训练模式，由于训练时会进行大量并行的简单运算，此时GPU处理起来效率会更高，因此我们需要了解如何使用GPU训练模型。

#### (1) .cuda()调用GPU训练

这种方法语法上最简单，直接在需要更改GPU的地方加上.cuda即可，下面是修改的部分代码:

```python
# 假设model是一个训练好的vgg16模型

model = model.cuda()

loss_fn = loss_fn.cuda()

imgs = imgs.cuda()

targets = targets.cuda()
```

即我们只需要修改：

模型、损失函数、input和targets即可。

#### (2) 设置训练device调用GPU训练

这种方法，有全局变量的思想，更容易在后期进行调节：

```python
# 假设model是一个训练好的vgg16模型

device = torch.device("cuda")

model = model.to(device)

loss_fn = loss_fn.to(device)

imgs = imgs.to(device)

targets = targets.to(device)
```

#### (3) 注意事项

经过GPU训练的模型，我们在使用时，**需要把我们的input转换为cuda**，否则无法正常使用：

```python
# 假设model是一个训练好的vgg16模型

input_data = input_data.cuda()

output = model(input_data)
```

### 4.谷歌colab简介

谷歌为我们提供了一个在线训练模型的网站：colab，可以使用谷歌的云服务器，从而体验高配置的训练过程：

colab的地址：https://colab.research.google.com/

colab使用方法：

首先，要登录Google账户

之后点击文件，选择新建笔记本，即可开始编写Python代码

![image-20230711150432499](C:\Users\Lobster\AppData\Roaming\Typora\typora-user-images\image-20230711150432499.png)

开启GPU优化：

![image-20230711150531043](C:\Users\Lobster\AppData\Roaming\Typora\typora-user-images\image-20230711150531043.png)

查看当前云服务器的配置：

```python
print("============查看GPU信息================")
# 查看GPU信息
!/opt/bin/nvidia-smi
print("==============查看pytorch版本==============")
# 查看pytorch版本
import torch
print(torch.__version__)
print("============查看虚拟机硬盘容量================")
# 查看虚拟机硬盘容量
!df -lh
print("============查看cpu配置================")
# 查看cpu配置
!cat /proc/cpuinfo | grep model\ name
print("=============查看内存容量===============")
# 查看内存容量
!cat /proc/meminfo | grep MemTotal

```

## 第四章: 尾声：用我们训练好的模型来输出结果

训练好模型以后，我们可以开始用训好的模型来做一些简单的实物测试：

首先，我们任选一种方法加载我们的模型：

```python
import torch

model = torch.load("trained_model.pth")
```

接下来，我们加载一张图片用来实物测试：

```python
from PIL import Image

img = Image.open("test.jpg")
```

由于我们的模型对input的图像格式有特殊的要求，需要用resize和reshape进行调整：

```python
from torchvision import transforms

compose = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32,32))
    ])

img = compose(img)

img = img.reshape(img,(1,3,32,32))
```

注意，此时看似可以直接把img丢进model中，但是前面说过，如果是用GPU训练的模型，需要把input进行cuda转换：

```python
img = img.cuda()
```

之后即可丢进模型：

```python
output = model(img)
```

但是我们的output，对于分类问题，会给出一个one-hot向量，其中向量的每个值代表属于某一个类型的概率，我们可以对output进行转换，使其直接输出最高的概率对应的类型的编号：

```python
output = output.argmax(1)
```

这样output通过**argmax**，就把原来一行概率向量，转化为代表分类结果的向量了。举一个例子：

output = [0.1,0.2,0.3]

通过argmax：output = [2]

**argmax(),参数0代表纵向，1代表横向。**

这样我们就拿到了预测的结果。

到此，pytorch就算入门了，但是后面的路还很长...