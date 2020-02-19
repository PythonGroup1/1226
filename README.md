import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./', one_hot=True)

# 输入数据input[?,784]

# 特征数量
num_features = 784

n_hidden_1 = 128#第一隐含层的神经元的数量

n_hidden_2 = 128#第二层

n_hidden_3 = 256#第三层

n_hidden_4 = 256#第四层

n_hidden_5 = 512#第二隐含层的神经元的数量

out_classes = 10#输出层，10类手写数字的结果

# Store layers weight & bias

# A random value generator to initialize weights.
# 正太分布的数据
random_normal = tf.initializers.Random_normal()

weights = {
    'h1': tf.Variable(random_normal([num_features, n_hidden_1])),
    'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(random_normal([n_hidden_4, n_hidden_5])),
    'out': tf.Variable(random_normal([n_hidden_5, out_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'b4': tf.Variable(tf.zeros([n_hidden_4])),
    'b5': tf.Variable(tf.zeros([n_hidden_5])),
    'out': tf.Variable(tf.zeros([out_classes]))
}

#建构神经网络模型
# 5层隐含层
# 深度神经网络：隐含层多一些，结构和我们现在所构建一样
def neural_net(x):
    # 第一层隐含层 128 神经元 就是矩阵运算
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # 激活函数，sigmoid函数
    layer_1 = tf.nn.sigmoid(layer_1)
    
    # 第二层隐含层 128 神经元 也是矩阵运算
    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
    # 激活函数，sigmoid.
    layer_2 = tf.nn.relu(layer_2)
    
        # 第三层隐含层 256 神经元 也是矩阵运算
    layer_3 = tf.matmul(layer_2, weights['h3']) + biases['b3']
    # 激活函数，sigmoid.
    layer_3 = tf.nn.relu(layer_3)
    
        # 第四层隐含层 256 神经元 也是矩阵运算
    layer_4 = tf.matmul(layer_3, weights['h4']) + biases['b4']
    # 激活函数，sigmoid.
    layer_4 = tf.nn.relu(layer_4)
    
        # 第五层隐含层 512 神经元 也是矩阵运算
    layer_5 = tf.matmul(layer_4, weights['h5']) + biases['b5']
    # 激活函数，sigmoid.
    layer_5 = tf.nn.relu(layer_5)
    
    # 输出层，也就是，预测的结果，10分类的问题。矩阵运算，变成10分类问题
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    # 矩阵运算的结果，转化成概率.
    return tf.nn.softmax(out_layer)
    
    # 交叉熵.
def cross_entropy(y_pred, y_true):
    # 数据裁剪，防止log(0).
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # 计算交叉熵.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),axis = -1))

# 计算准确率.
def accuracy(y_pred, y_true):
    # y_pred算法计算，返回数据类型是，tensor
    y1 = y_pred.numpy().argmax(axis = -1)
    y2 = y_true.argmax(axis = 1)
    return (y1 == y2).mean()

# 随机梯度下降.
optimizer = tf.keras.optimizers.SGD(0.01)

# 优化过程. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        y_pred = neural_net(x)
        loss = cross_entropy(y_pred, y)
        
    # 声明变量时，使用字典（方便管理），变量6个，计算梯度时，合并到一个列表中
    trainable_variables = list(weights.values()) + list(biases.values())

    # 计算梯度.[h1,h2,out_w,b1,b2,out_b]
    gradients = g.gradient(loss, trainable_variables)
    
    # 更新变量W和b，根据梯度进行更新.
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
for step in range(10001):
    X_train,y_train = mnist.train.next_batch(500)
    run_optimization(X_train, y_train)
    
    if step % 100 == 0:
        X_test,y_test = mnist.test.next_batch(500)
        y_pred = neural_net(X_test)
        loss = cross_entropy(y_pred, y_test)
        acc = accuracy(y_pred, y_test)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
