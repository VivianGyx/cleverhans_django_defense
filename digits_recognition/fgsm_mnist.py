"""
Use fast gradient sign method to craft adversarial on MNIST.

Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from attacks import fgm
import digits_recognition.model_attack as model


img_size = 28
img_chan = 1
n_classes = 10

#加载数据
print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255

X_test = [X_test[0]]

#现在我想要生成我上传图片的对抗样本
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical  #将类矢量（整数）转换为二进制类矩阵。
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting model')   #分割数据

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')    #建设图


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')
    env.x_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x_adv')

    env.ybar, logits = model.model(env.x, logits=True, training=env.training)
    env.ybar_adv, logits_adv = model.model(env.x_adv, logits=True, training=env.training)


    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        count_adv = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar_adv, axis=1))
        env.acc_adv = tf.reduce_mean(tf.cast(count_adv, tf.float32), name='acc_adv')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        xent_adv = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits_adv)
        env.loss = tf.reduce_mean(xent, name='loss')

        env.loss_adv = tf.reduce_mean(xent_adv, name='loss_adv')

        env.loss_fgsm = 0.5*(env.loss+env.loss_adv)


    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)
        env.train_op_Adv = optimizer.minimize(env.loss_fgsm)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    env.x_fgsm = fgm(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)
    env.ybarAdv, logitsAdv = model.model(env.x_fgsm, logits=True, training=env.training)


print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

def evaluate(sess, env, X_adv, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        if X_adv is not None:
            batch_loss, batch_acc = sess.run(
                [env.loss_fgsm, env.acc_adv],
                feed_dict={
                           env.x_adv: X_adv[start:end],
                           env.x: X_data[start:end],
                           env.y: y_data[start:end]})
        else:
            batch_loss, batch_acc = sess.run(
                [env.loss, env.acc],
                feed_dict={env.x: X_data[start:end],
                           env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, X_adv=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        # if shuffle:
        #     print('\nShuffling model')
        #     ind = np.arange(n_sample)
        #     np.random.shuffle(ind)
        #     X_data = X_data[ind]
        #     y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            if X_adv is not None:
                sess.run(env.train_op_Adv, feed_dict={
                                                  env.x: X_data[start:end],
                                                  env.x_adv: X_adv[start:end],
                                                  env.y: y_data[start:end],
                                                  env.training: True})

            else:
                sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                                  env.y: y_data[start:end],
                                                  env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_adv, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    过运行env.ybar进行推理。
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv

print('\nGenerating adversarial model')
# X_adv = make_fgsm(sess, env, X_train, eps=0.02, epochs=12)

# np.save('npy/fgsm_train1', X_adv)
X_adv = np.load('npy/fgsm_train.npy')

print('\nTraining')
train(sess, env, X_train, y_train, X_valid, y_valid, X_adv, load=False, epochs=5,
      name='fgsm1')

print('\nGenerating adversarial model')
X_adv1 = make_fgsm(sess, env, X_test, eps=0.02, epochs=12)

print('\nEvaluating on adversarial model')
evaluate(sess, env, X_adv1, X_test, y_test)

# print('\nRandomly sample adversarial model from each category')
#
# y1 = predict(sess, env, X_test)
# y2 = predict(sess, env, X_adv)
#
# print(y1)
#
# z0 = np.argmax(y_test, axis=1)
# z1 = np.argmax(y1, axis=1)
# z2 = np.argmax(y2, axis=1)
#
# X_tmp = np.empty((10, 28, 28))
# y_tmp = np.empty((10, 10))
# for i in range(10):
#     print('Target {0}'.format(i))
#     ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
#     cur = np.random.choice(ind)
#     X_tmp[i] = np.squeeze(X_adv[cur])
#     y_tmp[i] = y2[cur]
#
# print('\nPlotting results')
#
# fig = plt.figure(figsize=(10, 1.2))
# gs = gridspec.GridSpec(1, 10, wspace=0.05, hspace=0.05)
#
# label = np.argmax(y_tmp, axis=1)
# proba = np.max(y_tmp, axis=1)
# for i in range(10):
#     ax = fig.add_subplot(gs[0, i])
#     ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
#                   fontsize=12)
#
# print('\nSaving figure')
#
# gs.tight_layout(fig)
# os.makedirs('img', exist_ok=True)
# plt.savefig('img/fgsm_mnist.png')
