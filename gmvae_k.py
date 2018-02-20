import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
from gmvae_model import GMVAE
from utils import plot_r_by_c_images, load_and_mix_data, plot_scatter

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

try:
    k = int(sys.argv[1])
except IndexError:
    k=2
    print('Setting default value k={0}'.format(k))

dataset = load_and_mix_data('generated_from_cluster',k)

model = GMVAE(k=k,n_x=2)

saver = tf.train.Saver(keep_checkpoint_every_n_hours=4)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Restore parameters from file (optional)
    saver.restore(sess, './savedModels/2018-2-20/model-30')
    #model.test_stuff(sess, dataset)

    # TRAINING
    sess_info = (sess, saver)
    #model.train('logs/gmvae_k={:d}.log'.format(k), dataset, sess_info, epochs=30)

    # SCATTER PLOT

    zm0, zm1 = sess.run([model.zm[0],model.zm[1]], feed_dict={'x:0':dataset.test.data})
    labels = np.argmax(dataset.test.labels,axis=1)
    zm = np.concatenate((zm0[labels == 0],zm1[labels == 1]))
    plot_scatter(zm,labels,'scatter_predicted_zm.png')

    x0, x1 = sess.run([model.px_logit[0], model.px_logit[1]],
                        feed_dict={'x:0': dataset.test.data})
    x = np.concatenate((x0[labels == 0], x1[labels == 1]))
    plot_scatter(x, labels, 'scatter_predicted_x.png')
