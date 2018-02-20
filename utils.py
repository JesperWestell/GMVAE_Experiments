from tensorbayes.utils import progbar
import tensorflow as tf
import numpy as np
from scipy.stats import mode
import numpy as np
import os.path
import datetime
import matplotlib.pyplot as plt
from dataset import Dataset

def plot_scatter(X, Y, file_name):
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.set_title(file_name)
    sub.set_xlabel('x1')
    sub.set_ylabel('x2')
    for i, row in enumerate(X):
        sub.plot(row[0], row[1], '+'
                 , color="blue" if Y[i] == 0 else "red")

    plt.savefig(file_name)
    plt.show()
    print('scatter plot drawn')

def load_and_mix_data(fname,k):
    all_data = []
    all_labels = []
    for i in range(k):
        data = np.load('./generatedData/{0}{1}.npy'.format(fname,i))
        all_data.append(data)
        all_labels.append(np.full((data.shape[0],1),i))

    concatenated_data = np.concatenate(all_data, axis=0)
    concatenated_labels = np.concatenate(all_labels, axis=0)
    m = concatenated_data.shape[0]

    indices = np.random.permutation(np.arange(len(concatenated_data)))
    train_indices = indices[:int(m * 0.9)]
    test_indices = indices[int(m * 0.9):]

    train_data = concatenated_data[train_indices]
    train_labels = concatenated_labels[train_indices]
    test_data = concatenated_data[test_indices]
    test_labels = concatenated_labels[test_indices]
    dataset = Dataset()
    dataset.setTrainData(train_data, train_labels)
    dataset.setTestData(test_data, test_labels)
    return dataset

def get_var(name):
    all_vars = tf.global_variables()
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def save_params(saver,sess,epoch):
    now = datetime.datetime.now()
    base = './savedModels/'
    folder = '{0}-{1}-{2}/'.format(now.year, now.month, now.day)
    name = 'model'
    save_path = base+folder+name
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

def train(fname, mnist, sess_info, epochs):
    (sess, qy_logit, nent, loss, train_step, saver) = sess_info
    f = open_file(fname)
    iterep = 500
    for i in range(iterep * epochs):
        sess.run(train_step, feed_dict={'x:0': mnist.train.next_batch(100)[0]})

        progbar(i, iterep)
        if (i + 1) %  iterep == 0:
            a, b = sess.run([nent, loss], feed_dict={'x:0': mnist.train.images[np.random.choice(50000, 10000)]})
            c, d = sess.run([nent, loss], feed_dict={'x:0': mnist.test.images})
            a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
            e = test_acc(mnist, sess, qy_logit)
            string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
            stream_print(f, string, i <= iterep)
            string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, c, d, e, int((i + 1) / iterep)))
            stream_print(f, string)
        # Saves parameters every 5 epochs
        if (i + 1) % (5*iterep) == 0:
            print('saving')
            save_params(saver,sess,(i+1)//iterep)
    if f is not None: f.close()

def plot_r_by_c_images(images,r=10,c=10):
    print(r,c)
    canvas = np.empty((28 * r, 28 * c))
    for x in range(c):
        for y in range(r):
            canvas[y * 28:(y+1) * 28,
            x * 28:(x + 1) * 28] = images[y*c+x]
    plt.figure()
    plt.imshow(canvas, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    plt.savefig('mnist', bbox_inches='tight')
    plt.show()


