import numpy as np
import random
import traceback
import scipy.io as sio

from preprocess import preprocess
from deepnet import deepnet
from checkgrad import checkgrad
from initweights import initweights
from compute_loss import compute_loss


def example_tests():
    random.seed(31415926535)

    r = 0
    ok = 0
    s = []

    bostonData = sio.loadmat('./boston.mat')
    xTr = bostonData['xTr']
    yTr = bostonData['yTr']
    xTe = bostonData['xTe']

    wst = np.array([1, 12, np.shape(xTr)[0]])
    w = initweights(wst)

    print('Starting Test 1')
    try:
        addon = ''
        xTr, _, u, m = preprocess(xTr, xTe)
        failtest = np.int8(np.sum(np.linalg.svd(np.cov(xTr))[1])) != xTr.shape[0]
    except:
        failtest = True
        addon = traceback.format_exc()

    if failtest:
        r = r + 1
        s += 'Failed Test 1 incorrect preprocessing.\n' + addon + '\n'
    else:
        ok = ok + 1

    print('Completed Test 1\n')

    print('Starting Test 2')
    try:
        addon = ''
        graderr = checkgrad(deepnet, w, 1e-05, xTr, yTr, wst)
        failtest = graderr > 1e-02
    except:
        failtest = True
        addon = traceback.format_exc()

    if failtest:
        r = r + 1
        s += 'Failed Test 2 incorrect deepnet does not pass checkgrad.\n' + addon + '\n'
    else:
        ok = ok + 1

    print('Completed Test 2\n')

    yTr = np.array([0, 0, 1, 1, 0]).reshape(1, -1)
    zs = [np.array([.2, .2, 1.2, 1.2, .2]).reshape(1, -1)]
    print('Starting Test 3')
    try:
        addon = ''
        loss = compute_loss(zs, yTr)
        failtest = np.abs(loss - 0.02) > 1e-02
    except:
        failtest = True
        addon = traceback.format_exc()

    if failtest:
        r = r + 1
        s += 'Failed Test 3 incorrect loss computed.\n' + addon + '\n'
    else:
        ok = ok + 1

    print('Completed Test 3\n')

    return r, ok, s


if __name__ == '__main__':
    failed, ok, msgs = example_tests()
    print("Number of failed example tests: " + str(failed))
    print("Number of passed example tests: " + str(ok))
    if len(msgs):
        failMsg = 'Unfortunately, you failed %d test(s) on this evaluation: \n\n' % len(msgs)
        messages = ''
        for j in range(0, len(msgs)):
            messages += msgs[j]
        print(messages)
    # print("\nNote: we only implemented 3 out of 12 tests for you. "
    #       "Check the inline documentation for what the other tests do and implement them yourself!")
