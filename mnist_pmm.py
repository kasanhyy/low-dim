import numpy as np

from pmm import pmm_data
from private_PCA import proj_pmm_data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow.keras.datasets.mnist as mnist28
from sklearn import svm

import matplotlib.pyplot as plt
from plot import computeW1, plotW1


def mnist8data():
    """
    :return: The dataset of handwritten digits of size 8*8
    """
    mnist = load_digits()
    x, test_x, y, test_y = train_test_split(mnist.data, mnist.target, test_size=0.25,
                                            random_state=np.random.randint(1000))
    return x / 16, y, test_x / 16, test_y


def mnistUCI():
    """
    :return: The dataset of handwritten digits of size 8*8, larger amount
    """
    with open('optical+recognition+of+handwritten+digits/optdigits.tra') as fp:
        lines = fp.readlines()
        lines = [line.split(',') for line in lines]
        x = np.array([line[:-1] for line in lines], dtype='float') / 16
        y = np.array([line[-1] for line in lines], dtype='int')
    with open('optical+recognition+of+handwritten+digits/optdigits.tes') as fp:
        lines = fp.readlines()
        lines = [line.split(',') for line in lines]
        test_x = np.array([line[:-1] for line in lines], dtype='float') / 16
        test_y = np.array([line[-1] for line in lines], dtype='int')
    return x, y, test_x, test_y


def mnist28data():
    """
    :return: The dataset of handwritten digits of size 28*28
    """
    (x_train, y_train), (x_test, y_test) = mnist28.load_data()  # 载入数据
    x_train, x_test = x_train / 255.0, x_test / 255.0  # 3. 数据归一化， 范围0到1之间，因为像素值的范围是0~255
    x_train = np.reshape(x_train, (len(x_train), 784))
    x_test = np.reshape(x_test, (len(x_test), 784))
    return x_train, y_train, x_test, y_test


def unlabeledSynData(x, y=np.zeros(1), isLowDim=1, eps=1, max_size=10000, d2=4, label=-1):
    """
    Generate synthetic data with non-labelled original data, or for a particular label
    :param x: original data
    :param y: labels
    :param isLowDim: Boolean for whether to use low-dim algorithm
    :param eps: privacy parameter
    :param max_size: the size of data if there's enough
    :param d2: lower dimension
    :param label: -1 if non-labelled, or indicate the specific label
    :return: (the actual size, synthetic data)
    """
    if not y.any():
        label = -1
    if label != -1:
        x = np.array([x[i] for i in range(len(x)) if y[i] == label])

    n = min(len(x), max_size)

    if isLowDim:
        syndata = proj_pmm_data(x[:n], eps=eps, d2=d2)
    else:
        syndata = pmm_data(x[:n], eps)
    return n, syndata


def labeledSynData(x, y, isLowDim=1, eps=1, max_size=10000, d2=8):
    """
    Generate synthetic data with labelled data
    :param x: original data
    :param y: labels
    :param isLowDim: Boolean for whether to use low-dim algorithm
    :param eps: privacy parameter
    :param max_size: the size of data if there's enough
    :param d2: lower dimension
    :param label: -1 if non-labelled, or indicate the specific label
    :return: (the actual size, synthetic data)
    """
    n = min(len(x), max_size)

    data10 = [0] * 10
    for i in range(10):
        data10[i] = np.array([x[j] for j in range(n) if y[j] == i])

    syndata = [0] * 10
    for i in range(10):
        if isLowDim:
            temp = proj_pmm_data(data10[i], eps=eps, d2=d2)
        else:
            temp = pmm_data(data10[i], eps, scale=1)
        syndata[i] = temp
    return n, syndata


def plotUnlabeled(data, true_size, eps=1, A=5, B=5, level='simple'):
    """
    Plot the image of data
    :param data: synthetic data to be plotted
    :param true_size: the size of original data
    :param eps: privacy param
    :param A: num of rows of subgraphs
    :param B: num of cols of subgraphs
    :param level: 'simple' for  8*8, 'complex' for 28*28
    :return: None
    """
    n, d = data.shape
    fig, axs = plt.subplots(A, B)
    for i in range(A):
        for j in range(B):
            pic = data[np.random.randint(n)]
            newshape = (8, 8) if level == 'simple' else (28, 28)
            pic = np.reshape(pic, newshape)
            plt.subplot(A, B, A * i + j + 1)
            plt.axis('off')
            axs[i, j] = plt.imshow(pic, cmap='gray')

    # plt.axis('off')
    fig.suptitle(f'MNIST synthetic data, n={true_size}, d={d}, eps={eps}', size=16)
    plt.show()


def plotLabeled(data, true_size, eps=1, level='simple'):
    """
    Plot the image of data in labels with 4x5 many subgraphs
    :param data: synthetic data to be plotted
    :param true_size: the size of original data
    :param eps: privacy param
    :param level: 'simple' for  8*8, 'complex' for 28*28
    :return: None
    """
    # d = len(data[0][0])
    fig, axs = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            label = 5 * (i // 2) + j
            data_num = len(data[label])
            if data_num != 0:
                pic = data[label][np.random.randint(data_num)]
            else:
                continue
            newshape = (8, 8) if level == 'simple' else (28, 28)
            pic = np.reshape(pic, newshape)
            plt.subplot(4, 5, 5 * i + j + 1)
            plt.axis('off')
            axs[i, j] = plt.imshow(pic, cmap='gray')

    # plt.axis('off')
    fig.suptitle(f'n={true_size}, eps={eps}', size=16)
    plt.show()


def list2labels(data_list):
    """
    Change the list of data into one array
    :param data_list: [data0, data1, ..., data9]
    :return: array of data with labels
    """
    return np.concatenate(data_list), np.concatenate(
        [i * np.ones(len(data_list[i]), dtype=int) for i in range(10)])


def unit():
    """ Old unit, please use svm_unit()"""
    level = 'complex'  # complex -- 60000*28*28, simple -- 1300 * 8*8
    isLowDim = 0
    d2 = 4 if level == 'simple' else 4
    print(f'd\' = {d2}')
    max_size = 10000
    eps = 1

    if level == 'simple':
        x, y, test_x, test_y = mnistUCI()
    elif level == 'complex':
        x, y, test_x, test_y = mnist28data()

    n, syndata = labeledSynData(x, y, isLowDim, eps=eps, max_size=max_size, d2=d2)
    plotLabeled(syndata, n, eps, level)

    train_x, train_y = list2labels(syndata)
    model = svm.LinearSVC(max_iter=7000)
    model.fit(train_x, train_y)
    z = model.predict(test_x)
    print('准确率:', np.sum(z == test_y) / z.size)

    # n, syndata = unlabeledSynData(x, y, isLowDim, label=6, eps=eps, max_size=max_size, d2=d2)
    # plotUnlabeled(syndata, n, eps)

    # computeW1(x[:n], syndata, metric='chebyshev')
    # print()


def syndata_unit(level='simple',
                 isLowDim=1,
                 max_size=10000,
                 eps=1,
                 d2=4,
                 plot_img=0):
    """
    A computing unit for SVM_accuracy with the following customized parameters
    :param level: 'simple' for  8*8, 'complex' for 28*28
    :param isLowDim: Boolean, if true, using low-dim subroutine
    :param max_size: the size of data if there's enough
    :param eps: privacy parameter
    :param d2: lower dimension
    :param plot_img: Boolean, if true, plot the image of data
    :return: the SVM accuracy
    """
    if level == 'simple':
        x, y, test_x, test_y = mnistUCI()
    else:
        x, y, test_x, test_y = mnist28data()

    n, syndata = labeledSynData(x, y, isLowDim, eps=eps, max_size=max_size, d2=d2)
    if plot_img:
        plotLabeled(syndata, n, eps, level)

    syn_x, syn_y = list2labels(syndata)
    return x, y, syn_x, syn_y, test_x, test_y


def accuracy(x, y, syn_x, syn_y, test_x, test_y, accuracy_type='mean'):
    """
    :param x: original data
    :param y: original labels
    :param syn_x: synthetic data
    :param syn_y: synthetic labels
    :param test_x: test data
    :param test_y: test labels
    :param accuracy_type: string 'mean', 'cov', 'svm'
    :return: corresponding error for 'mean' and 'cov', accuracy rate for 'svm'
    """
    if accuracy_type == 'svm':
        return svm_accuracy(syn_x, syn_y, test_x, test_y)
    elif accuracy_type == 'cov':
        return cov_accuracy(x, y, syn_x, syn_y)
    elif accuracy_type == 'cov_fro':
        return cov_accuracy(x, y, syn_x, syn_y, order='fro')
    elif accuracy_type == 'mean':
        return mean_accuracy(x, y, syn_x, syn_y)
    elif accuracy_type == 'mean_l2':
        return mean_accuracy(x, y, syn_x, syn_y, order=2)
    else:
        raise ValueError('Accuracy type not supported')


def mean_accuracy(x, y, syn_x, syn_y, order=np.inf):
    """
    :return: the max mean error over all digits
    """
    n, d = x.shape
    n_syn = len(syn_x)

    mean_x = np.zeros((10, d))
    for i in range(n):
        mean_x[y[i]] += x[i]
    mean_x /= n

    mean_syn_x = np.zeros((10, d))
    for i in range(n_syn):
        mean_syn_x[syn_y[i]] += syn_x[i]
    mean_syn_x /= n_syn
    return np.max(np.linalg.norm(mean_x - mean_syn_x, axis=1, ord=order))


def cov_accuracy(x, y, syn_x, syn_y, order=None):
    """
    :return: the cov mean error over all digits and all X_i X_j
    """
    n, d = x.shape
    n_syn = len(syn_x)

    cov_x = np.zeros((10, d, d))
    for i in range(n):
        cov_x[y[i]] += np.outer(x[i], x[i])
    cov_x /= n

    cov_syn_x = np.zeros((10, d, d))
    for i in range(n_syn):
        cov_syn_x[syn_y[i]] += np.outer(syn_x[i], syn_x[i])
    cov_syn_x /= n
    if order is None:
        return np.max(np.abs(cov_x - cov_syn_x))
    else:
        return np.max(np.linalg.norm(cov_x - cov_syn_x, axis=(1,2), ord=order))


def svm_accuracy(train_x, train_y, test_x, test_y):
    """
    :return: the accuracy of the SVM task
    """
    model = svm.LinearSVC(max_iter=5000)
    model.fit(train_x, train_y)
    z = model.predict(test_x)
    return np.sum(z == test_y) / z.size


def multi_eps_plot(accuracy_type='mean'):
    """
    Plot the accuracy rate with eps in "eps_list". One experiment in original dimension and the others in "dims".
    eps_list:   the eps to be chosen
    n:          the number of data to analyze
    level:      'simple' for 8*8, 'complex' for 28*28
    iter:       num of repetition for accuracy
    dims:       the lower dimension levels, besides original dimension
    :return: None
    """

    # eps_list = np.array([0.5, 1, 2, 4, 8])
    # eps_list = np.logspace(-2, 2, 5)
    eps_list = np.array([0.1, 0.3, 1, 3, 10, 30])
    n = 10000
    level = 'simple'
    iter = 3

    dims = (2, 4, 8, 12)

    # simulate in original dimension
    y = []
    for eps in eps_list:
        print(f'eps={eps}, direct pmm')
        s = 0
        for i in range(iter):
            s += accuracy(*syndata_unit(level=level, max_size=n, eps=eps, isLowDim=0), accuracy_type)
            # pass
        s /= iter
        y.append(s)

    # simulate in lower dimensions in dims
    y2 = [[] for i in range(len(dims))]
    for i in range(len(dims)):
        for eps in eps_list:
            print(f'eps={eps}, d\'={dims[i]}')
            s = 0
            for j in range(iter):
                s += accuracy(*syndata_unit(level=level, max_size=n, eps=eps, d2=dims[i]), accuracy_type)
            s /= iter
            y2[i].append(s)

    # Plot the result of y and y2[:]
    colors = ('b', 'c', 'g', 'k', 'm', 'brown')
    x = range(len(eps_list))
    plt.plot(x, y, color='red', linewidth=1, linestyle='--', label='Direct PMM')
    for i in range(len(dims)):
        plt.plot(x, y2[i], color=colors[i], label=f'PMM d\'={dims[i]}')

    # Plot parameters
    topic = 'svm accuracy' if accuracy_type == 'svm' else accuracy_type + ' error'
    plt.xticks(x, eps_list)
    plt.xlabel('Epsilon')
    plt.ylabel(f'Average {topic}')
    if accuracy_type == 'svm':
        plt.ylim((0, 1))
    # elif accuracy_type in ('mean', 'cov'):
    #     plt.ylim((0, 0.4))
    else:
        pass
        # plt.ylim((0, 1))

    # The title and the name
    topic += ' n=3823' if level == 'simple' else f' n={min(n, 60000)}'
    plt.title(f'{topic}, d\'={dims}')
    plt.legend()
    if round(eps_list[1] / eps_list[0]) == 2:
        plt.savefig(f'image_res/{topic}, d2={dims}, eps=logspace(2)')
    elif round(eps_list[1] / eps_list[0]) == 3:
        plt.savefig(f'image_res/{topic}, d2={dims}, eps=logspace(3)')
    else:
        plt.savefig(f'image_res/{topic}, d2={dims}')
    plt.show()


if __name__ == '__main__':
    n = 10000
    level = 'simple'
    eps = 3
    datas = syndata_unit(level='simple', eps=eps, max_size=n, isLowDim=1, plot_img=1)
    print(f"True data accuracy {accuracy(*datas, accuracy_type='svm')}")

    # print(accuracy(*datas, accuracy_type='mean'))

    # multi_eps_plot(accuracy_type='cov_fro')
