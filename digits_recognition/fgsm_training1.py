"""
Use fast gradient sign method to craft adversarial on MNIST.

Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np


import tensorflow as tf

from attacks import fgm, pgd
import digits_recognition.model_attack as model


img_size = 28
img_chan = 1
n_classes = 10


print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255

# X_test = [X_test[0]]

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

    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    env.x_fgsm = fgm(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

    env.pgd_eps = tf.placeholder(tf.float32, (), name='pgd_eps')
    env.pgd_epsilon = tf.placeholder(tf.float32, (), name='pgd_epsilon')
    env.pgd_epochs = tf.placeholder(tf.int32, (), name='pgd_epochs')
    env.x_pgd = pgd(model, env.x, epsilon=env.pgd_epsilon, epochs=env.pgd_epochs, eps=env.pgd_eps, rand=True)

    env.training = tf.placeholder_with_default(False, (), name='mode')

    # 干净样本相关内容
    env.ybar, logits = model.model(env.x, logits=True, training=env.training)
    count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
    env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
    xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                 logits=logits)
    env.loss = tf.reduce_mean(xent, name='loss')
    optimizer = tf.train.AdamOptimizer()
    env.train_op = optimizer.minimize(env.loss)

    # 对抗样本相关内容
    env.ybar_adv, logits_adv = model.model(env.x_pgd, logits=True, training=env.training)
    count_adv = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar_adv, axis=1))
    env.acc_adv = tf.reduce_mean(tf.cast(count_adv, tf.float32), name='acc')
    xent_adv = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits_adv)
    env.loss2 = tf.reduce_mean(xent_adv, name='loss')
    env.loss_adv = 0.5 * (env.loss + env.loss2)
    optimizer = tf.train.AdamOptimizer()
    env.train_op_adv = optimizer.minimize(env.loss_adv)

    env.saver = tf.train.Saver()



print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=128):
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
        batch_loss, batch_acc = sess.run(
            [env.loss_adv, env.acc_adv],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end],
                       env.pgd_epsilon: 0.03,
                       env.pgd_eps: 0.1,
                       env.pgd_epochs: 12})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
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

        if shuffle:
            print('\nShuffling model')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op_adv, feed_dict={env.x: X_data[start:end],
                                                  env.y: y_data[start:end],
                                                  env.training: True,
                                                  env.pgd_epsilon: 0.03,
                                                  env.pgd_eps: 0.1,
                                                  env.pgd_epochs: 12})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

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


def make_pgd(sess, env, X_data, epsilon=0.3, epochs=1, eps=0.01, batch_size=128):
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
        adv = sess.run(env.x_pgd, feed_dict={
            env.x: X_data[start:end],
            env.pgd_epsilon: epsilon,
            env.pgd_eps: eps,
            env.pgd_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


# print('\nTraining')
#
# train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
#       name='mnist')

# print('\nEvaluating on clean model')
#
# evaluate(sess, env, X_test, y_test)

# print('\nGenerating adversarial model')

# X_adv = make_fgsm(sess, env, X_train, eps=0.02, epochs=12)
# X_adv = make_pgd(sess, env, X_train, eps=0.02, epochs=12)

# print('\nEvaluating on adversarial model')
#
# evaluate(sess, env, X_adv, y_test)


# X_adv1 = make_fgsm(sess, env, X_train, eps=0.02, epochs=12)
# np.save('npy/fgsm_train.npy', X_adv1)
# X_adv2 = make_pgd(sess, env, X_train, eps=0.02, epochs=12)
# np.save('npy/pgd_train.npy', X_adv2)

# X_adv1 = np.load('npy/fgsm_test.npy')

# print(type(X_train))
# X_adv2 = np.load('npy/pgd_train.npy')
# print(type(X_adv2))
# D = np.vstack((X_adv2, X_train))
#
# np.save('npy/pgd_and_clean_train.npy', D)

# X_adv_and_clean = np.load('npy/pgd_and_clean_train.npy')

# print(X_adv_and_clean.shape)


# y_adv_and_clean = np.vstack((y_train, y_train))

# print(y_adv_and_clean.shape)

# train(sess, env, X_adv1, y_train, X_valid, y_valid, load=True, epochs=5,
#       name='fgsm_mnist')
# train(sess, env, X_adv1, y_train, X_valid, y_valid, load=True, epochs=5,
#       name='pgd_mnist')
print('\nTraining')


train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
      name='pgd_attack1')
# X_adv = np.load("npy/pgd_test.npy")
# X_adv1 = make_pgd(sess, env, X_adv, eps=0.02, epochs=12)
#
# evaluate(sess, env, X_test, y_test)
