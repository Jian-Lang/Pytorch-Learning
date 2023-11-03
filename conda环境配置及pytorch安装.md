## **Windows安装配置conda环境 + 安装cuda版本pytorch**

前言：安装顺序依次为：安装anaconda - - - > 通过anaconda创建Python环境 - - - > 在对应的环境中安装pytorch - - - > 安装pycharm - - - > 在pycharm中配置conda环境

### 1. 安装anaconda并配置环境变量

这部分内容完全按照下面这篇CSDN博文执行：

https://blog.csdn.net/m0_61607990/article/details/129531686

### 2. 在anaconda中创建Python环境

anaconda安装好，环境也配好之后，打开anaconda prompt，而后通过下面的指令创建新的环境(区别于base环境)：

```bash
conda create -n 虚拟环境名称 python=版本号
```

名字建议起Pytorch，这个环境是pytorch的基环境，后续的其他项目环境可以从这个环境上通过克隆命令进行克隆(这是后话了)：

```bash
conda create --name <new_name> --clone <old_name>
```

### 3. 在新建的环境中安装pytorch

首先，进入新建的环境(按环境名为Pytorch)：

```bash
conda activate Pytorch
```

而后，先安装几个机器学习必备的库(非pytorch)：

**numpy**

```bash
pip install numpy -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple
```

**pandas**

```bash
pip install pandas -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple
```

**matplotlib**

```bash
pip install matplotlib -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple
```

**tqdm**

```bash
pip install tqdm
```

**sklearn**

```bash
pip install scikit-learn -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple 
```

**pytorch**

之后，到pytorch的官网上，找到适合自己电脑cuda版本的pytorch安装指令(通常最新即可)

下面指令可以辅助查看自己电脑的cuda版本

```bash
nvidia-smi
```

这里给出记录此过程时的最新cuda版pytorch安装指令：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

这里要注意的是，安装pytorch不可以换源，否则安装的版本会有各种bug

### 4. pycharm中配置anaconda环境

最后，在pycharm中配置anaconda环境即可：

![image-20231102210134628](C:\Users\ADMIN\AppData\Roaming\Typora\typora-user-images\image-20231102210134628.png)

![image-20231102210442135](C:\Users\ADMIN\AppData\Roaming\Typora\typora-user-images\image-20231102210442135.png)

### 5.验证效果

在新项目中，复制下面的代码，运行：

```Python
import torch # 如果pytorch安装成功即可导入

print(torch.cuda.is_available()) # 查看CUDA是否可用
```

如果打印出True，则说明 all is fune！

![image-20231102210647004](C:\Users\ADMIN\AppData\Roaming\Typora\typora-user-images\image-20231102210647004.png)

### 后续

具体新的项目，可以在这个Pytorch环境基础上，通过克隆环境，而后在具体项目的环境中安装一些独有的包即可。
