u"""
    勉強会用コード：Tensorflowを用いた二次元畳み込み処理を学ぶ(2)

    勉強用コードなのでコメント盛りだくさんです。
    あらかじめご了承ください。

    [謝辞] 以下の元コードを拡張したものである。
    https://github.com/nfmcclure/tensorflow_cookbook/blob/master/08_Convolutional_Neural_Networks/02_Intro_to_CNN_MNIST/02_introductory_cnn.py
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
import time
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# MNISTデータの読み込み処理。
# 読込形式だが、目的変数は one-hot形式ではない。
# /tmp/data に MNISTデータをキャッシュする。
data_dir = '/tmp/data'
mnist = input_data.read_data_sets(data_dir, one_hot=False)

# Convert images into 28x28 (they are downloaded as 1x784)
# 各画像の画素データを(28x28)の二次元に変換する。
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

# Convert labels into one-hot encoded vectors
train_labels = mnist.train.labels
test_labels = mnist.test.labels

# 学習バッチサイズ
batch_size = 100

# 学習率
learning_rate = 0.005

# 評価時のサンプル数
evaluation_size = 500

# MNIST画像の幅(pixel)
image_width = train_xdata[0].shape[0]

# MNIST画像の高さ(pixel)
image_height = train_xdata[0].shape[1]

# 目的変数の種類の数
target_size = np.max(train_labels) + 1

# グレースケールなので、チャンネル数は１である。
num_channels = 1

# 学習イテレーション数
generations = 500

# 5イテレーションごとに評価する
eval_every = 5

# 畳み込み第一層で生成する特徴量の数
conv1_features = 25

# 畳み込み第二層で生成する特徴量の数
conv2_features = 50

# 実験２
#conv1_features = 10
#conv2_features = 20

# 実験３
#conv1_features = 40
#conv2_features = 80

# Max-Pooling第一層のプーリングウィンドウサイズ
max_pool_size1 = 2  # NxN window for 1st max pool layer

# Max-Pooling第二層のプーリングウィンドウサイズ
max_pool_size2 = 2  # NxN window for 2nd max pool layer

# 全結合第一層の大きさ
fully_connected_size1 = 100

# プレースホルダの定義
x_input_shape = (batch_size, image_height, image_width, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size))
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape=(evaluation_size))

# 畳み込みフィルタのサイズ定義
# 第一、第二畳み込み層で共通
filter_height = 2
filter_width = 2

# 畳み込み第一層のフィルタの重みを定義する。
# 標準偏差 0.1, 平均 0 の正規乱数で初期化する。
conv1_weight = tf.Variable(tf.truncated_normal([filter_height, filter_width, num_channels, conv1_features],
                                               stddev=0.1, dtype=tf.float32))

# 畳み込み第一層の出力に加算するバイアスを定義する。
# 畳み込み第一層で生成される特徴量の数だけバイアスを用意するので、一次元でサイズは conv1_features となる。
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))


# 畳み込み第二層のフィルタの重みを定義する。
# こちらも標準偏差 0.1, 平均 0 の正規乱数で初期化する。
conv2_weight = tf.Variable(tf.truncated_normal([filter_height, filter_width, conv1_features, conv2_features],
                                               stddev=0.1, dtype=tf.float32))

# 畳み込み第二層の出力に加算するバイアスを定義する。
# 畳み込み第二層で生成される特徴量の数だけバイアスを用意するので、一次元でサイズは conv2_features となる。
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))


# 全結合層の定義
resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)

# [全結合第一層]
# 一次元に均して入力するので、
# 入力サイズは、畳み込み第二層の出力の幅、高さ、特徴量の数の積となる。
full1_input_size = resulting_width * resulting_height * conv2_features

# 全結合第一層には、[バッチサイズ, full1_input_size] の入力が入ってくる。
# 重み行列の行数は full1_input_size 行となる。
# 標準偏差0.1, 平均 0 の正規乱数で初期化する。
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1],
                                               stddev=0.1, dtype=tf.float32))

# 全結合第一層の出力に加えるバイアスを定義する。
# 各サンプルにつき (fully_connected_size1)個の要素の値が計算されているので、
# バイアスの形状も一次元で、[fully_connected_size1]となる。
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

# [全結合第二層]
# 全結合第一層の出力は、[バッチサイズ, fully_connected_size1] という形状になっている。、
# これを入力として、10個の数字のうちどれが最も近いのか数値を出力する。
# よって、形状は [fully_connected_size1, target_size] = [fully_connected_size1, 10] となる。
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size],
                                               stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))


# [ＣＮＮを構築する関数]
# 入力 conv_input_data が、形状[バッチサイズ, 28, 28, 1] であることを前提にしている。
def my_conv_net(conv_input_data):
    # First Conv-ReLU-MaxPool Layer
    conv1 = tf.nn.conv2d(conv_input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                               strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

    # Second Conv-ReLU-MaxPool Layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

    # 畳み込み第二層の出力(max_pool2)を、各サンプルごとに一次元に平坦化して
    # 全結合第一層に入力できるようにする。
    final_conv_shape = max_pool2.get_shape().as_list()
    # final_conv_shape[0] : バッチサイズが入っている。
    # final_conv_shape[1] : 各サンプルの高さ
    # final_conv_shape[2] : 各サンプルの幅
    # final_conv_shape[3] : 各サンプルの各ピクセルのカラーチャネル数
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    # 全結合第一層の定義
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

    # 全結合第二層の定義
    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

    return final_model_output


# 二つのＣＮＮを定義する。
# 学習用には model_output を使い、
# テスト用には test_model_outputを使う。
model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

# 損失関数の定義：
# ソフトマックス交差エントロピーを用いる。
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

# 予測関数の定義：
# 学習用ＣＮＮとテスト用ＣＮＮで分けて定義する。
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)


# 精度評価関数の定義：
# 予測値(ソフトマックス形式）logits と 正解 targets を渡して、精度を計算する。
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return 100. * num_correct / batch_predictions.shape[0]


# 最適化器の定義
# -> AdamOptimizerにする手もあるか。実験してみるべし。
my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

# 各 Variable の初期化。
# Tensorflow を使う上でのお約束である。
init = tf.global_variables_initializer()
sess.run(init)

# [学習ループの定義]
train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    # バッチサイズの分だけ、無作為にデータを抽出する。
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    rand_x = train_xdata[rand_index]

    # MNISTは、カラーチャネル情報を持っていないので、カラーチャネル次元を追加する。
    # 各ピクセルの画素がスカラー値から一次元配列になる。
    rand_x = np.expand_dims(rand_x, 3)

    rand_y = train_labels[rand_index]

    # プレースホルダにセットする。
    train_dict = {x_input: rand_x, y_target: rand_y}

    # 学習プロセスの実行
    # train_step は、学習用ＣＮＮを用いた 入力 -> 損失関数の計算 -> 最適化を順番に実行するTensorflowグラフである。
    sess.run(train_step, feed_dict=train_dict)

    # 学習時の性能を測定する。
    # loss : 学習用ＣＮＮを用いた 入力 -> 損失関数の計算 を実行する。
    # prediction : 学習用ＣＮＮを用いた 入力 -> 予測 を実行する。
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)

    # 学習用ＣＮＮの予測 temp_train_preds と 正解 rand_y を比較し、精度を求める。
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)

    if (i + 1) % eval_every == 0:
        # [評価処理]
        # テスト用ＣＮＮを使って、未知のデータに対する性能を測定する。

        # 評価用サンプル(500個) を無作為に抽出する。
        eval_index = np.random.choice(len(test_xdata), size=evaluation_size)
        eval_x = test_xdata[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]

        # 評価用プレースホルダに評価用サンプルをセットする
        test_dict = {eval_input: eval_x, eval_target: eval_y}

        # テスト用ＣＮＮを用いて 入力 -> 予測 を実行する。
        test_preds = sess.run(test_prediction, feed_dict=test_dict)

        # 評価用サンプルに対する予測 test_preds と正解 eval_y を使って、精度を求める。
        temp_test_acc = get_accuracy(test_preds, eval_y)

        # [性能記録処理:学習時の性能]
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)

        # [性能記録処理:テスト時の性能]
        test_acc.append(temp_test_acc)

        # 標準出力に性能情報（学習時＆テスト時）を出力する。
        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# ==========================
# 性能曲線をグラフに描画する
# ==========================

# 時系列インデックス : 性能測定をした学習世代番号のリストである
eval_indices = range(0, generations, eval_every)

# 損失値の時系列推移を描画する。
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()


# トレーニング精度とテスト精度の時系列推移を描画する。
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# [予測と正解の比較]
# 無作為に６つのサンプルを取り出し、予測と正解の一致・不一致を画像で描画する。
actuals = rand_y[0:6]
predictions = np.argmax(temp_train_preds, axis=1)[0:6]
images = np.squeeze(rand_x[0:6])

Nrows = 2
Ncols = 3
for i in range(6):
    plt.subplot(Nrows, Ncols, i + 1)
    plt.imshow(np.reshape(images[i], [28, 28]), cmap='Greys_r')
    plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
              fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
plt.show()


# [畳み込みフィルタの重みの視覚化]
# conv1_weight を画像に変換する。
# 重み行列を [特徴量, 高さ, 幅, チャンネル番号] の順番に次元を入れ替えた後、
# 次元数が１だけのチャンネル番号の次元がなくなり、
# learned_weight は、[特徴量, 高さ, 幅] の三次元になる。
learned_weight = sess.run(tf.squeeze(tf.transpose(conv1_weight,perm=[3,0,1,2])))
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
print(learned_weight.shape)
node_size, filter_height, filter_width = learned_weight.shape
for i in range(node_size):
    ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(learned_weight[i], cmap="gray", interpolation='nearest')

plt.show()

time.sleep(1000)