from tensorbayes.utils import progbar
import tensorflow as tf
import numpy as np
from scipy.stats import mode
import numpy as np
import os.path
import datetime
import matplotlib.pyplot as plt
from dataset import Dataset

### PLOTTING HELPER FUNCTIONS ###

def plot_z_means(sess, X, Y, model, k, n_z):
    '''
        Given examples data, computes and plots the mean of the examples latent
        variables
    '''
    all_zm = np.zeros((len(X), k, n_z))
    for i in range(k):
        all_zm[:, i] = sess.run(model.zm[i],
                                feed_dict={'x:0': X})

    qy = sess.run(model.qy, feed_dict={'x:0': X})
    y_pred = one_hot(qy.argmax(axis=1), depth=k).astype(bool)

    zm = all_zm[y_pred]
    labels = np.argmax(Y, axis=1)
    plot_labeled_data(zm, labels, 'scatter_predicted_zm.png')

def plot_z(sess, X, Y, model, k, n_z):
    '''
        Given examples data, computes and plots their latent variables
    '''
    all_z = np.zeros((len(X), k, n_z))
    for i in range(k):
        all_z[:, i] = sess.run(model.z[i],
                                feed_dict={'x:0': X})

    qy = sess.run(model.qy, feed_dict={'x:0': X})
    y_pred = one_hot(qy.argmax(axis=1), depth=k).astype(bool)

    z = all_z[y_pred]
    labels = np.argmax(Y, axis=1)
    plot_labeled_data(z, labels, 'scatter_predicted_z.png')

def plot_gmvae_output(sess, X, Y, model, k):
    '''
        Given examples data, computes and plots the output of the examples
    '''
    x_dims = len(X[0])
    all_x = np.zeros((len(X), k, x_dims))

    for i in range(k):
        all_x[:, i] = sess.run(model.px_logit[i],
                               feed_dict={'x:0': X})

    qy = sess.run(model.qy, feed_dict={'x:0': X})
    y_pred = one_hot(qy.argmax(axis=1), depth=k).astype(bool)

    x = all_x[y_pred]
    labels = np.argmax(Y, axis=1)
    plot_labeled_data(x, labels, 'scatter_predicted_x.png')

def sample_z(sess, model, y, num_samples=6):
    '''
        Given a gaussian category, sample latent variables from that gaussian
        distribution
    '''
    # Need to feed x to get proper shape of input
    zm, zv = sess.run([model.zm_prior[y], model.zv_prior[y]],
                      feed_dict={'x:0': np.zeros((num_samples, model.n_x))})
    z_sample = np.random.normal(loc=zm, scale=np.sqrt(zv))
    return z_sample

def sample_x(sess, model, y=None, num_samples=6):
    '''
        Given a gaussian category, sample latent variables from that gaussian
        distribution, and compute their variable representations
    '''
    z = sample_z(sess, model, y, num_samples=num_samples)
    x = sess.run(model.px_logit[y],
                 feed_dict={
                     'graphs/hot_at{:d}/qz/z_sample:0'.format(y): z})
    return x

def sample_and_plot_z(sess, k, model, num_samples):
    """
    For all gaussian categories, sample and plot latent variables
    """
    sample_list = []
    for i in range(k):
        sample_list.append(sample_z(sess, model, i, num_samples))
    plot_data_clusters(sample_list, 'generated_z.png')

def sample_and_plot_x(sess, k, model, num_samples):
    """
        For all gaussian categories, sample latent variables and plot
        their variables representations
    """
    sample_list = []
    for i in range(k):
        sample_list.append(sample_x(sess, model, i, num_samples))
    plot_data_clusters(sample_list, 'generated_x.png')

def plot_data_clusters(clusters, file_name):
    colors = ['green','cyan','magenta']
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.set_title(file_name)
    sub.set_xlabel('x1')
    sub.set_ylabel('x2')
    for i, X in enumerate(clusters):
        color = colors[i]
        for i, row in enumerate(X):
            sub.plot(row[0], row[1], '+', color=color)

    plt.savefig(file_name)
    plt.show()
    print('scatter plot drawn')

def plot_labeled_data(X, Y=None, file_name=None):
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.set_title(file_name)
    sub.set_xlabel('x1')
    sub.set_ylabel('x2')
    for i, row in enumerate(X):
        if type(Y) is np.ndarray:
            color = "blue" if Y[i] == 0 else "red"
        else:
            color = 'green'
        sub.plot(row[0], row[1], '+', color=color)

    plt.savefig(file_name)
    plt.show()
    print('scatter plot drawn')

def plot_r_by_c_images(images, r=10, c=10):
    '''
    Plots MNIST images
    '''
    print(r, c)
    canvas = np.empty((28 * r, 28 * c))
    for x in range(c):
        for y in range(r):
            canvas[y * 28:(y + 1) * 28,
            x * 28:(x + 1) * 28] = images[y * c + x]
    plt.figure()
    plt.imshow(canvas, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    plt.savefig('mnist', bbox_inches='tight')
    plt.show()

########################################

def one_hot(labels, depth):
    one_hot = np.zeros((len(labels), depth))
    labels = np.reshape(labels, (len(labels, )))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def load_and_mix_data(fname, k, randomize=True):
    all_data = []
    all_labels = []
    for i in range(k):
        data = np.load('./generatedData/{0}{1}.npy'.format(fname, i))
        all_data.append(data)
        all_labels.append(np.full((data.shape[0], 1), i))

    concatenated_data = np.concatenate(all_data, axis=0)
    concatenated_labels = np.concatenate(all_labels, axis=0)
    m = concatenated_data.shape[0]
    if randomize:
        indices = np.random.permutation(np.arange(len(concatenated_data)))
    else:
        indices = np.arange(len(concatenated_data))
    train_indices = indices[:int(m * 0.9)]
    test_indices = indices[int(m * 0.9):]

    train_data = concatenated_data[train_indices]
    train_labels = concatenated_labels[train_indices]
    test_data = concatenated_data[test_indices]
    test_labels = concatenated_labels[test_indices]
    dataset = Dataset(k)
    dataset.setTrainData(train_data, train_labels)
    dataset.setTestData(test_data, test_labels)
    return dataset

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def save_params(saver, sess, epoch):
    now = datetime.datetime.now()
    base = './savedModels/'
    folder = '{0}-{1}-{2}/'.format(now.year, now.month, now.day)
    name = 'model'
    save_path = base + folder + name
    saver.save(sess, save_path, global_step=epoch)

def test_acc(dataset, sess, qy_logit):
    # Basically, for each category, look at the most common digit among the test
    # examples in that category, and predict all those test examples as that
    # most common digit
    logits = sess.run(qy_logit, feed_dict={'x:0': dataset.test.data})
    cat_pred = logits.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    real_labels = dataset.test.labels.argmax(1)
    for cat in range(logits.shape[1]):
        idx = cat_pred == cat
        lab = real_labels[idx]
        if len(lab) == 0:
            continue
        real_pred[idx] = mode(lab).mode[0]
    return np.mean(real_pred == real_labels)

def open_file(fname):
    if fname is None:
        return None
    else:
        i = 0
        while os.path.isfile('{:s}.{:d}'.format(fname, i)):
            i += 1
        return open('{:s}.{:d}'.format(fname, i), 'w', 1)

def get_var(name):
    all_vars = tf.global_variables()
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None