import tensorflow as tf

INPUTSIZE = 7
HIDDEN1SIZE = 10
HIDDEN2SIZE = 8
OUTPUTSIZE = 5

Learning_Rate = 0.05

x = tf.placeholder(dtype = tf.float32, shape = [None,INPUTSIZE])
y_ = tf.placeholder(dtype = tf.float32, shape = [None,OUTPUTSIZE])

inputData = [[1,5,3,7,8,10,12]
            ,[5,8,10,3,9,7,1]
            ,[6,4,32,2,3,6,1]
            ,[3,6,3,9,5,3,2]
            ,[8,5,3,1,6,3,6]]
label = [[0,0,0,1,0]
        ,[1,0,0,0,1]
        ,[0,1,1,0,0]
        ,[0,1,0,0,0]
        ,[0,0,1,0,1]]

tensor_map = {x: inputData, y_: label}

#신경망 구축

W_h1 = tf.Variable( tf.truncated_normal(shape=[INPUTSIZE, HIDDEN1SIZE]), dtype = tf.float32)
b_h1 = tf.Variable( tf.zeros([HIDDEN1SIZE]), dtype = tf.float32)

W_h2 = tf.Variable( tf.truncated_normal(shape=[HIDDEN1SIZE, HIDDEN2SIZE]), dtype = tf.float32)
b_h2 = tf.Variable( tf.zeros([HIDDEN2SIZE]), dtype = tf.float32)

W_o = tf.Variable( tf.truncated_normal(shape=[HIDDEN2SIZE, OUTPUTSIZE]), dtype = tf.float32)
b_o = tf.Variable( tf.zeros([OUTPUTSIZE]), dtype = tf.float32)

#저장
param_list = [W_h1, b_h1, W_h2, b_h2, W_o, b_o]
saver =tf.train.Saver(param_list)


hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + b_h1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + b_h2)
y = tf.sigmoid(tf.matmul(hidden2, W_o) + b_o)

#트레이닝 사전 정의

cost = tf.reduce_sum(-y_*tf.log(y) - (1-y_)*tf.log(1-y), axis=[1,0])
cost - tf.reduce_mean(cost)
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess = tf.Session()

init = tf.global_variables_initializer()
#sess.run(init)
#불러오기
saver.restore(sess, './tensorflow_live.ckpt')

#트레이닝

for i in range(1000):
    loss = sess.run([train, cost], feed_dict = tensor_map)
    correct = sess.run(tf.cast(correct_prediction, tf.float32), tensor_map)
    pred = sess.run(accuracy, tensor_map)
    result = sess.run(y, tensor_map)
    if i % 100 == 0 :
        saver.save(sess, './tensorflow_live.ckpt')
        print("step=",i)
        print("loss=",loss)
        print("pred=", pred)
        print("correct=", correct_prediction)
        print("result=", result)
#result = sess.run(y, feed_dict = {x: [[4,7,3,2,7,9,2]]})
#print(result)

sess.close()
