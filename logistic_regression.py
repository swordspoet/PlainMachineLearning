from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集，loadDataSet()函数的主要功能是打开文本文件，并逐行读取；
# 每行的前两列分别为X1,X2，除此以外，还为偏置项设置一列X0
def loadDataSet():
    data = []; label = []
    fr = open('E:/Python/data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        data.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label.append(int(lineArr[2]))
    return data, label
data, label = loadDataSet()

# 定义sigmoid函数
def sigmoid(z):
    return 1.0 / (1 + exp(-z))

# 定义梯度下降算法
# 设定步长alpha为0.001，迭代次数为500次，初始权重theta为长度为n个值全为1的向量
def gradAscent(data, label):
    dataMatrix = np.matrix(data); labelMatrix = np.matrix(label).T
    m, n = shape(dataMatrix)
    alpha = 0.001
    iters = 500
    theta = ones((n, 1))
    for k in range(iters):
		# 梯度下降算法，因为要求损失函数的最小值
		# 对应公式（12）
        h = sigmoid(dataMatrix * theta)
        error = (h - labelMatrix)
        theta = theta - alpha * dataMatrix.T * error
    return theta

theta = gradAscent(data, label)
theta = array(theta)
dataArr = array(data)

# 画出决策边界
def plotfit(theta):
    n = dataArr.shape[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x1.append(dataArr[i,1]); y1.append(dataArr[i,2])
        else:
            x2.append(dataArr[i,1]); y2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s = 30, c = 'red', marker = 'o')
    ax.scatter(x2, y2, s = 30, c = 'blue', marker = 'x')
	# 创建等差数列，设定x的取值范围为-3.0到3.0
    x = arange(-3.0, 3.0, 0.1)
    y = (-theta[0] - theta[1] * x) / theta[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
plotfit(theta)

# 随机梯度下降（SGD）

def stoGradDscent0(dataMatrix, labelMatrix):
    m, n = shape(dataMatrix)
    alpha = mat([0.01])
    theta = mat(ones(n)) #ones()函数是numpy中的一个组件，在使用之前必须先进行 from numpy import *操作，否则会出现错误“ones is not defined”
    for i in range(n):
        h = sigmoid(sum(dataMatrix[i] * theta.T))
        error = labelMatrix[i] - h
        theta = theta + alpha * error * dataMatrix[i]
    return theta

theta = stoGradDscent0(data, label)

def plotfit1(theta):
    import matplotlib.pyplot as plt
    import numpy as np
    dataArr = array(dataMatrix)
    n = dataArr.shape[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x1.append(dataArr[i,1]); y1.append(dataArr[i,2])
        else:
            x2.append(dataArr[i,1]); y2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s = 30, c = 'red', marker = 'o')
    ax.scatter(x2, y2, s = 30, c = 'blue', marker = 'x')
    x = arange(-3.0, 3.0, 0.1)  #创建等差数列
    y = (-theta[0,0] - theta[0,1] * x) / theta[0,2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

plotfit1(theta)
# 第一次优化的效果不佳，有差不多三分之一的点被误分类，为此进行第二次算法优化

def stoGradDscent1(dataMatrix, labelMatrix, iters = 150):
    m, n = shape(dataMatrix)
    theta = mat(ones(n))
    for j in range(iters):
        dataIndex = range(m)  # m = 100
        for i in range(m):
			# 因为alpha在每次迭代的时候都会调整，这可以缓解数据的波动，alpha会减小，但不会到零
			# 通过随机选取样本来更新回归系数，这种方法可以减少周期性的波动
			# uniform()方法将随机生成下一个实数，它在[x,y]范围内
            alpha = 4 / (1.0+j+i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*theta.T))
            error = label[randIndex] - h
            theta = theta + alpha * error * dataMatrix[randIndex]
    return theta

theta = stoGradDscent1(dataMatrix, labelMatrix, iters = 150)
theta

plotfit1(theta)