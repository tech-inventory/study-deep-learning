u"""
    CNN 学習用コード
    y.takahashi

    畳み込み層、ReLu層、Max-Pooling層の動作を確認するための
    学習用コードである。

    学習用のため、コメント多いのはご了承あれ。

    [謝辞] 以下のコードを参考にさせていただいた。
    https://github.com/nfmcclure/tensorflow_cookbook/blob/master/08_Convolutional_Neural_Networks/02_Intro_to_CNN_MNIST/02_introductory_cnn.py

"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# MNISTのデータセットを読み込む。
# one_hot = True とすると、目的変数(=正解の数字)を one hot形式で読み込む。
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# mnist.train.images には、トレーニングセットの画素データが格納されている。
# ただし、784個の一次元配列の形式なので、それを (28,28) の二次元に形状を変換しておく。
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

# 試しに１個のトレーニングセットの画像を表示してみる。
# (28,28) の二次元配列になっていることが分かるはず。
print(train_xdata[0])

# 畳み込みの処理を確認する。
#
sess = tf.Session()

# グレースケールなので、チャンネル数は1
num_channels = 1

# バッチサイズは100
batch_size = 100

# 畳み込み層で生成する特徴量の数。
# 以下の場合、１画素あたり、25個の特徴量が生成される。
conv1_features = 25

# Max-Poolingで用いるウィンドウのサイズ。
# 2 x 2 (pixel)のサイズを定義している。
max_pool_size1 = 2

# 画像サイズをデータセットから取得する。
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]

# プレースホルダの定義：
# Tensorflow の場合、畳み込み層への入力変数の形状は (バッチサイズ, 高さ, 幅, カラーチャンネル数)というお約束になっている。
# よって、その形状でプレースホルダを定義する必要がある。
x_input_shape = (batch_size, image_height, image_width, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)

# 畳み込み層のフィルタを定義する。
# conv1_weight がフィルタの定義である。
# 今回の場合、4 x 4 pixel サイズを各チャンネルに適用し、それを25件分実行し、、
# 各pixelごとに25個の特徴量を生成する。
conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features],
                                               stddev=0.1, dtype=tf.float32))

# ストライド = [1,1,1,1] なので、X,Yともに1pixelずつ移動させて、畳み込み計算を実施する。
# ただし、サンプル番号(第１次元）、チャンネル（第４次元）のストライドに1以外を適用することはできない。
# padding = 'SAME' なので、畳み込みフィルタが画像の範囲外を含んだ場合、0 で埋め尽くす。
conv1 = tf.nn.conv2d(x_input, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')

# 畳み込み計算の結果にバイアス値 conv1_bias を加算して
# ReLu 活性化関数に通す。これにより、負の値は全て0にクリッピングされる。
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

# Max-Pooling層の定義。
# Poolingのウィンドウサイズは max_pool_size1 x max_pool_size1 で、
# max_pool_size1 x max_pool_size1 の範囲内で最大の値を抽出する。
# ストライドは [1, max_pool_size1, max_pool_size1, 1] なので、Max-Poolingを適用した領域が
# オーバーラップしないようにPoolingウィンドウを走査する。
# よって、(28,28) の大きさのものが (14,14)に縮小されることになる。
max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                           strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

# ============================================================================================
# ここから Tensorflow 上で畳み込み層、ReLu層、Max-Pooling層の動きを確かめる。
# ============================================================================================
init = tf.global_variables_initializer()
sess.run(init)

# トレーニング画像セットから、バッチサイズの数だけ無作為にサンプルを抽出する。
rand_index = np.random.choice(len(train_xdata), size=batch_size)
rand_x = train_xdata[rand_index]

# トレーニング画像セットは、(28,28)の二次元なので、このままでは畳み込み層に入力することができない。
# そこで、チャンネル次元を追加する。
rand_x = np.expand_dims(rand_x, 3)

# feed_dict 用の定義。プレースホルダに画像サンプルを入力する。
train_dict = {x_input: rand_x}

# 畳み込み層の計算結果を取り出す。
conv1_result = sess.run(conv1, feed_dict=train_dict)
print(rand_x[0])
print(conv1_result[0])

# ReLu層の結果を取り出す。
relu1_result = sess.run(relu1, feed_dict=train_dict)
print(relu1_result[0])

# ReLu層を通す前と、通した後の値を比較する。
print("***** ReLu層を通す前 *****")
print(conv1_result[0][11][16])
print("***** ReLu層を通した後 *****")
print(relu1_result[0][11][16])

# Max-Pooling層の結果を取り出す。
max_pool1 = sess.run(max_pool1, feed_dict=train_dict)
print(max_pool1[0])
print("*** debug ***")

