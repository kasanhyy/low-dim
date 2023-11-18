
from pca_10sep import proj_pmm_data
from mnist_pmm import mnist28data, mnist8data, plotLabeled, list2labels
from sklearn import svm
import numpy as np


if __name__ == '__main__':
    level = 'simple'
    x, y, test_x, test_y = mnist28data() if level == 'complex' else mnist8data()

    d2 = 2
    print(f'd\' = {d2}')
    max_size = 10000
    eps = 1

    n = min(len(x), max_size)
    syndata = proj_pmm_data(x[:n], y, eps=eps, d2=d2)
    plotLabeled(syndata, n, eps, level)

    train_x, train_y = list2labels(syndata)
    model = svm.LinearSVC(max_iter=5000)
    model.fit(train_x, train_y)
    z = model.predict(test_x)
    print('准确率:', np.sum(z == test_y) / z.size)
