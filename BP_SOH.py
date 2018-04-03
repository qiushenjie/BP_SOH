import numpy as np 
import pandas as pd
import tensorflow as tf 
from matplotlib import pyplot as plt

Path = 'SOH_Data.xlsx'

#训练集
R1 = pd.read_excel(Path, sheetname=0)
R2 = pd.read_excel(Path, sheetname=1)
X_All = np.array(R1)
Y_All = np.array(R2)
x_train = X_All.astype('float64')
y_train = Y_All.astype('float64')

#测试集
R3 = pd.read_excel(Path, sheetname=2)
R4 = pd.read_excel(Path, sheetname=3)
X_All2 = np.array(R3)
Y_All2 = np.array(R4)
x_test = X_All2.astype('float64')
y_test = Y_All2.astype('float64')

# 定义tensorflow的session
sess = tf.Session()
X = tf.placeholder(tf.float64, name = 'X_input')
y_true = tf.placeholder(tf.float64, name = 'y_True')

#定义层数及神经元个数
IHO = [10, 8, 5, 4, 1]
L_IHO = len(IHO)

#初始化权重、偏置
Weight = []
for i in range(0, L_IHO - 1):
	Weight.append(tf.Variable(tf.zeros([IHO[i], IHO[i + 1]], tf.float64), name = 'Weight_%d'%(i + 1)))
Bias = []
for i in range(0, L_IHO - 1):
	Bias.append(tf.Variable(tf.zeros([IHO[i + 1]], tf.float64), name = 'Bais_%d'%(i + 1)))

#定义神经网络每层的运算方式： 运算=（输入*权重+偏置），再做非线性转换（sigmoid）
y_in = [] #模型输入
y_out = [] #模型输出
for i in range(0, L_IHO - 1): 
#模型算法， function部分
	if i == 0:
		y_in.append(tf.add(tf.matmul(X, Weight[i]), Bias[i], name ='y_input_%d'%(i + 1)))
		y_out.append(tf.sigmoid(y_in[i], name = 'y_output_%d'%(i + 1)))
	else:
		y_in.append(tf.add(tf.matmul(y_out[i - 1], Weight[i]), Bias[i], name ='y_input_%d'%(i + 1)))
		y_out.append(tf.sigmoid(y_in[i], name = 'y_output_%d'%(i + 1)))
y = y_out[-1] #取输出的最后一个元素

#定义损失函数，求损失
loss = tf.reduce_sum(tf.square(y - y_true), name = 'loss')

#构建优化器，最小化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss) #定义训练模式为最小损失值，学习率为0.01

#初始化整个网络
init = tf.global_variables_initializer() #初始化所有符号共享变量
sess = tf.Session()#构建会话
sess.run(init) #运行会话

#训练神经网络
Y = []
Loss = []
#循环2000次
for i in range(0, 200):
	sess.run(train_step, feed_dict = {X:x_train, y_true:y_train})
	if(i + 1)%100 == 0:
		curr_loss = sess.run(loss, feed_dict = {X:x_train, y_true:y_train})
		print('%4d:'%(i + 1), "loss:%s\n"%curr_loss)
		Y.append(sess.run(y_out, feed_dict = {X:x_train, y_true:y_train}))
	Loss.append([i, sess.run(loss, feed_dict = {X:x_train, y_true:y_train})])


#绘制模型输出的数据图表
W = sess.run(Weight, feed_dict = {X:x_train, y_true:y_train})
Y_Output = sess.run(y, feed_dict = {X:x_train, y_true:y_train})
Loss = np.array(Loss)
plt.plot(Loss[:, 0], Loss[:, 1])
plt.show()

#对测试集进行测定，并最小化均方差
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict = {X:x_test, y_true:y_test}))







