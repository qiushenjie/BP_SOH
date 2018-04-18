#电池老化率测定的神经网络模型
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'SOH_Data.xlsx'
#训练集读取及归一化
xTrainData = pd.read_excel(path, sheetname = 0)
yTrainData = pd.read_excel(path, sheetname = 1)
n1 = np.shape(xTrainData)[1]
x_data = np.array(xTrainData).astype('float32')
for i in range(n1):
    x_data[:, i] = (x_data[:, i] - np.amin(x_data[:, i]))/(np.amax(x_data[:, i]) - np.amin(x_data[:, i]))
y_data = np.array(yTrainData).astype('float32')
y_data[:] = (y_data[:] - np.amin(y_data[:]))/(np.amax(y_data[:]) - np.amin(y_data[:]))

#测试集读取及归一化
xTestData = pd.read_excel(path, sheetname = 2)
yTestData = pd.read_excel(path, sheetname = 3)
xTest = np.array(xTestData).astype('float32')
n2 = np.shape(xTrainData)[1]
xTrain = np.array(xTrainData).astype('float32')
for i in range(n2):
    xTest[:, i] = (xTest[:, i] - np.amin(xTest[:, i]))/(np.amax(xTest[:, i]) - np.amin(xTest[:, i]))
yTest = np.array(yTestData).astype('float32')
yTest[:] = (yTest[:] - np.amin(yTest[:]))/(np.amax(yTest[:]) - np.amin(yTest[:]))

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)#平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)#标准差
            tf.summary.scalar('max', tf.reduce_max(var))#最大值
            tf.summary.scalar('min', tf.reduce_min(var))#最小值
            tf.summary.histogram('histogram', var)#直方图

#5层神经网络，每层神经元个数
IHO = [12, 8, 5, 4, 1]

#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 12], name = 'xInput')
    y = tf.placeholder(tf.float32, [None, 1], name = 'y')

#神经元中间层
with tf.name_scope('layer'):
    with tf.name_scope('weights_L1'):
        Weight_L1 = tf.Variable(tf.random_normal([12, 8]), name = 'W1')
        variable_summaries(Weight_L1)
    with tf.name_scope('bias_L1'):
        biases_L1 = tf.Variable(tf.zeros([8]), name = 'b1')
        variable_summaries(biases_L1)
    with tf.name_scope('L_1'):
        Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biases_L1
        L1 = tf.nn.tanh(Wx_plus_b_L1)

    with tf.name_scope('weights_L2'):
        Weight_L2 = tf.Variable(tf.random_normal([8, 5]), name = 'W2')
        variable_summaries(Weight_L2)
    with tf.name_scope('bias_L2'):
        biases_L2 = tf.Variable(tf.zeros([5]), name = 'b2')
        variable_summaries(biases_L2)
    with tf.name_scope('L_2'):
        Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biases_L2
        L2 = tf.nn.tanh(Wx_plus_b_L2)

    with tf.name_scope('weights_L3'):
        Weight_L3 = tf.Variable(tf.random_normal([5, 4]), name = 'W3')
        variable_summaries(Weight_L3)
    with tf.name_scope('bias_L3'):  
        biases_L3 = tf.Variable(tf.zeros([4]), name = 'b3')
        variable_summaries(biases_L3)
    with tf.name_scope('L_3'):
        Wx_plus_b_L3 = tf.matmul(L2, Weight_L3) + biases_L3
        L3 = tf.nn.tanh(Wx_plus_b_L3)
#神经元输出层
    with tf.name_scope('weights_L4'):
        Weight_L4 = tf.Variable(tf.random_normal([4, 1]), name = 'W4')
        variable_summaries(Weight_L4)
    with tf.name_scope('bias_L4'):
        biases_L4 = tf.Variable(tf.zeros([1]), name = 'b4')
        variable_summaries(biases_L4)
    with tf.name_scope('prediction'):
        Wx_plus_b_L4 = tf.matmul(L3, Weight_L4) + biases_L4
        prediction = tf.nn.tanh(Wx_plus_b_L4)

#二次代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - prediction), name = 'loss')
    tf.summary.scalar('loss', loss)
#使用梯度下降法训练
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#合并所有summary
merged = tf.summary.merge_all()
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for i in range(10000):
        summary, _ = sess.run([merged, train_step], feed_dict = {x: x_data, y: y_data})
        writer.add_summary(summary, i)
        curr_loss = sess.run(loss, feed_dict = {x: x_data, y: y_data})
        if (i + 1)%100 == 0:
            print('第%d次迭代loss:'%(i + 1), curr_loss)
    #训练集预测集
    prediction_value = sess.run(prediction, feed_dict = {x: x_data})
    #测试集预测集
    prediction_value_test = sess.run(prediction, feed_dict = {x: xTest})
    test_loss = sess.run(loss, feed_dict = {x: xTest, y: yTest})
    print('测试误差：', test_loss)
    print(prediction_value_test)
