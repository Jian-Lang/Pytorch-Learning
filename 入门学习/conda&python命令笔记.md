## **Conda命令笔记**

## 预备式

执行conda指令之前，需要**先打开anaconda prompt**，而后输入指令

## **1. 创建/克隆新的conda虚拟环境：**

```bash
conda create -n 虚拟环境名称 python=版本号
```

```bash
conda create --name <new_name> --clone <old_name>
```

## 2. 进入某个虚拟环境:

```bash
conda activate 虚拟环境名称
```

## **3. 删除某个虚拟环境：**

```bash
conda remove -n lang_jian --all
```

## 4.列出当前所有虚拟环境(并用*指示当前所在环境)：

```bash
conda env list
```

## 5.列出环境下的所有库：

```bash
conda list
```

## **6. 安装/卸载三方库指令：**

```bash
conda install/remove xxx
```

## 7. 列出当前的所有源

```bash
conda config --show channels
```

## 8. 删除某个源

```bash
conda config --remove channels 源名称
```

## 9. 显示下载源

```bash
conda config --set show_channel_urls yes
```

p.s. 对源的操作，亦可从C/USER文件夹下的.condarc文件中修改(有些人在/.conda文件夹下)

pps. 新建环境时，要以管理员身份运行anaconda prompt。如果报错，可以先前往先前创建的环境中(通过activate语句)，之后再新建，另外镜像源中的forge也是有可能导致新建环境错误的

## Python命令笔记

## 1.安装换源指令

```
pip install x -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple

pip install x -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install x -i http://mirrors.aliyun.com/pypi/simple/

pip install x -i http://pypi.mirrors.ustc.edu.cn/simple/
```

