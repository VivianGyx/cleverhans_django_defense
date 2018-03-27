import numpy as np
import tensorflow as tf


from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils_mnist import data_mnist
from cleverhans import utils_tf, utils, attacks
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans.attacks import FastGradientMethod, MadryEtAl

# Get MNIST test data
train_start = 0
train_end = 60000
test_start = 0
test_end = 10000
X_train, y_train, X_test, y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                              test_end=test_end)

img_size = 28
img_chan = 1
n_classes = 10

fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.
               }
pgd_params = {'eps': 0.3,
              'eps_iter': .01,
              'clip_min': 0.,
              'clip_max': 1.}
train_params_clean = {
        'nb_epochs': 6,
        'batch_size': 128,
        'learning_rate': 0.001,
        'filename': 'clean_model',
        'train_dir': 'model_test_clean/'
}
train_params_adv = {
        'nb_epochs': 6,
        'batch_size': 128,
        'learning_rate': 0.001,
        'filename': 'attack_model',
        'train_dir': 'pgd_model/'
}

x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')

report = utils.AccuracyReport()

rng = np.random.RandomState([2017, 8, 30])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def train(sess, predictions_adv=True, nb_filters=64, batch_size=128):
    if predictions_adv:
        model_2 = make_basic_cnn(nb_filters=nb_filters)
        preds_2 = model_2(x)
        # fgsm2 = FastGradientMethod(model_2, sess=sess)
        # adv_x_2 = fgsm2.generate(x, **fgsm_params)
        pgd2 = MadryEtAl(model_2, sess=sess)
        adv_x_2 = pgd2.generate(x, **pgd_params)
        preds_2_adv = model_2(adv_x_2)

        def evaluate_2():
            # Accuracy of adversarially trained model on legitimate test inputs
            eval_params = {'batch_size': batch_size}
            accuracy = model_eval(sess, x, y, preds_2, X_test, y_test,
                                  args=eval_params)
            print('Test accuracy on legitimate examples: %0.4f' % accuracy)
            report.adv_train_clean_eval = accuracy

            # Accuracy of the adversarially trained model on adversarial examples
            accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                                  y_test, args=eval_params)
            print('Test accuracy on adversarial examples: %0.4f' % accuracy)
            report.adv_train_adv_eval = accuracy

        # Perform and evaluate adversarial training
        model_train(sess, x, y, preds_2, X_train, y_train, save=True,
                    predictions_adv=preds_2_adv, evaluate=evaluate_2,
                    args=train_params_adv, rng=rng)
    else:
        model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)

        model_train(sess, x, y, preds, X_train, y_train, save=True, evaluate=evaluate,
                    args=train_params_clean, rng=rng)


train(sess, predictions_adv=True, nb_filters=64, batch_size=128)