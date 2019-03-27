import numpy as np
import operator


def gen_bs_statistics(data, stat, n):
    results = np.empty(n)
    for i in range(n):
        bs = np.random.choice(data, size=len(data))
        results.append(stat(bs))
    return results


def gen_perm_statistics(data1, data2, stat, n):
    results = np.empty(n)
    data = np.concatenate((data1, data2))
    for i in range(n):
        perm = np.random.permutation(data)
        sample1 = perm[:len(data1)]
        sample2 = perm[len(data1):]
        results.append(stat(sample1, sample2))
    return results


def p_value(actual, reps, ge=True):
    comp = operator.ge if ge else operator.le
    return np.sum(comp(reps, actual)) / len(reps)
