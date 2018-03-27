from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import json
import digits_recognition.model_attack as model
from attacks import fgm, pgd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils_mnist import data_mnist
from cleverhans import utils_tf, utils, attacks
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans.attacks import FastGradientMethod


#加载数据
# print('\nLoading MNIST')
# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# Get MNIST test data
train_start = 0
train_end = 60000
test_start = 0
test_end = 10000
X_train, y_train, X_test, y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                              test_end=test_end)

# Use label smoothing
# assert y_train.shape[1] == 10
label_smooth = .1
y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)

img_size = 28
img_chan = 1
n_classes = 10
# Define input TF placeholder
x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')
X_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
# training = tf.placeholder_with_default(False, (), name='mode')
# eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
# epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')

train_params1 = {
        'nb_epochs': 6,
        'batch_size': 128,
        'learning_rate': 0.001,
        'filename': 'clean_model',
        'train_dir': 'model/'
    }
train_params2 = {
        'nb_epochs': 6,
        'batch_size': 128,
        'learning_rate': 0.001,
        'filename': 'attack_model',
        'train_dir': 'model/'
    }

fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}

pgd_params = {'eps': 0.3,
              'eps_iter': .01,
              'clip_min': 0.,
              'clip_max': 1.}


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


model_cnn = make_basic_cnn()
preds = model_cnn.get_probs(x)

fgsm = attacks.FastGradientMethod(model_cnn, sess=sess)
adv_fgsm = fgsm.generate(x, **fgsm_params)

pgd = attacks.MadryEtAl(model_cnn, sess=sess)
adv_pgd = pgd.generate(x, **pgd_params)

# model_2 = make_basic_cnn()
# preds_2 = model_2(x)
# fgsm2 = attacks.FastGradientMethod(model_2, sess=sess)
# adv_x_2 = fgsm2.generate(x, **fgsm_params)
# preds_2_adv = model_2(adv_x_2)

# Object used to keep track of (and return) key accuracies
    #用于跟踪（和返回）键精度的对象
report = utils.AccuracyReport()

rng = np.random.RandomState([2017, 8, 30])

base_url = 'digits_recognition/static/img/'


def evaluate(batch_size=128):  # 如果属实，请完成单元测试的准确性报告以验证性能是否足够
    # Evaluate the accuracy of the MNIST model on legitimate test
    # examples  评估MNIST模型在合法测试样本上的准确性
    eval_params = {'batch_size': batch_size}
    acc = utils_tf.model_eval(  # X_test: numpy array with training inputs； Y_test: numpy array with training outputs
        sess, x, y, preds, X_test, y_test, args=eval_params)
    report.clean_train_clean_eval = acc
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate examples: %0.4f' % acc)


def load(sess, name='model'):
    print('\nLoading saved model')
    tf_model_load(sess, 'digits_recognition/pgd_model/{}'.format(name))


load(sess, name='attack_model')


def predict(sess, X_data, batch_size=128):

    print('\nPredicting')
    n_classes = preds.get_shape().as_list()[1]
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(preds, feed_dict={x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, X_data, batch_size=128):
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
        adv = sess.run(adv_fgsm, feed_dict={
            x: X_data[start:end]
            })
        X_adv[start:end] = adv

    return X_adv


def make_pgd(sess, X_data, batch_size=128):
    """
    Generate PGD by running pgd.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(adv_pgd, feed_dict={
            x: X_data[start:end]
            })
        X_adv[start:end] = adv

    return X_adv


def index(request):
    return render(request, 'index.html')


def img_change(img):
    X_tmp1 = np.empty((10, 28, 28))
    X_tmp = 1 - img
    X_tmp1[0] = np.squeeze(X_tmp)
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(X_tmp1[0], cmap='gray', interpolation='none')
    # 去除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    # 去除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 设置大小
    fig.set_size_inches(2, 2)
    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    # plt.tight_layout()


@csrf_exempt
def process(request):
    # 标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)

    # 一维数组，输出10个预测概率
    output_clean = predict(sess, input).tolist()

    X_adv_fgsm = make_fgsm(sess, input)
    output_fgsm = predict(sess, X_adv_fgsm).tolist()

    X_adv_pgd = make_pgd(sess, input)
    output_pgd = predict(sess, X_adv_pgd).tolist()

    return HttpResponse(json.dumps([output_clean[0], output_fgsm[0], output_pgd[0]]))


@csrf_exempt
def drawInput(request):
    #标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    # print(input)
    img_change(input)
    plt.savefig(base_url + 'clean.png', bbox_inches="tight", pad_inches=0)
    return HttpResponse()


@csrf_exempt
def fgsm_attack(request):
    #标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    X_adv = make_fgsm(sess, input)
    img_change(X_adv)
    plt.savefig(base_url + 'fgsm_mnist.png', bbox_inches="tight", pad_inches=0)
    return HttpResponse()

@csrf_exempt
def pgd_attack(request):
    #标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    X_adv = make_pgd(sess, input)
    img_change(X_adv)
    plt.savefig(base_url + 'pgd_mnist.png', bbox_inches="tight", pad_inches=0)
    return HttpResponse()




