"""pyBKT 1.4.3's E-step, compiled.

pyBKT's pure-Python fallback walks every sequence and every time step in Python on each EM iteration, which takes
hours on EdNet. ``inner`` below is the same forward-backward pass, statement for statement and in the same order;
only the dot products of a 2x2 matrix with a 2-vector are written out. ``EM_fit.run`` looks ``inner`` up by name
at call time, so swapping it changes the speed of the fit and of ``predict``, not its arithmetic (checked against
the original in ``tests/test_kt.py``).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def _inner(As, Bn, initial_distn, allresources, starts, lengths, num_resources, num_subparts, alldata,
           start, end, bigT):
    N_R, N_S = 2 * num_resources, 2 * num_subparts
    trans = np.zeros((2, N_R))
    emission = np.zeros((2, N_S))
    init = np.zeros((2, 1))
    alpha_out = np.zeros((2, bigT))
    loglike = 0.0
    for sequence_index in range(start, end):
        s = starts[sequence_index] - 1
        T = lengths[sequence_index]
        likelihoods = np.ones((2, T))
        alpha = np.empty((2, T))
        for t in range(min(2, T)):
            for n in range(num_subparts):
                d = alldata[n, s + t]
                if d:
                    c = 2 * n + (1 if d == 2 else 0)
                    for i in range(2):
                        sl = Bn[i, c]
                        likelihoods[i, t] *= 1.0 if sl == 0 else sl
        alpha[0, 0] = initial_distn[0] * likelihoods[0, 0]
        alpha[1, 0] = initial_distn[1] * likelihoods[1, 0]
        norm = alpha[0, 0] + alpha[1, 0]
        alpha[0, 0] /= norm
        alpha[1, 0] /= norm
        loglike += np.log(norm)
        if T >= 2:
            k = 2 * (allresources[s] - 1)
            a0, a1 = alpha[0, 0], alpha[1, 0]
            alpha[0, 1] = (As[0, k] * a0 + As[0, k + 1] * a1) * likelihoods[0, 1]
            alpha[1, 1] = (As[1, k] * a0 + As[1, k + 1] * a1) * likelihoods[1, 1]
            norm = alpha[0, 1] + alpha[1, 1]
            alpha[0, 1] /= norm
            alpha[1, 1] /= norm
            loglike += np.log(norm)
        for t in range(2, T):
            for n in range(num_subparts):
                d = alldata[n, s + t]
                if d:
                    c = 2 * n + (1 if d == 2 else 0)
                    for i in range(2):
                        sl = Bn[i, c]
                        likelihoods[i, t] *= 1.0 if sl == 0 else sl
            k = 2 * (allresources[s + t - 1] - 1)
            a0, a1 = alpha[0, t - 1], alpha[1, t - 1]
            alpha[0, t] = (As[0, k] * a0 + As[0, k + 1] * a1) * likelihoods[0, t]
            alpha[1, t] = (As[1, k] * a0 + As[1, k + 1] * a1) * likelihoods[1, t]
            norm = alpha[0, t] + alpha[1, t]
            alpha[0, t] /= norm
            alpha[1, t] /= norm
            loglike += np.log(norm)

        gamma = np.empty((2, T))
        gamma[0, T - 1] = alpha[0, T - 1]
        gamma[1, T - 1] = alpha[1, T - 1]
        f = True
        pair = np.empty((2, 2))
        for t in range(T - 2, -1, -1):
            k = 2 * (allresources[s + t] - 1)
            a0, a1 = alpha[0, t], alpha[1, t]
            for i in range(2):
                pair[i, 0] = As[i, k] * a0
                pair[i, 1] = As[i, k + 1] * a1
            dotted0 = As[0, k] * a0 + As[0, k + 1] * a1
            dotted1 = As[1, k] * a0 + As[1, k + 1] * a1
            g0, g1 = gamma[0, t + 1], gamma[1, t + 1]
            pair[0, 0] = (pair[0, 0] * g0) / dotted0
            pair[1, 0] = (pair[1, 0] * g1) / dotted1
            pair[0, 1] = (pair[0, 1] * g0) / dotted0
            pair[1, 1] = (pair[1, 1] * g1) / dotted1
            for i in range(2):
                for j in range(2):
                    v = pair[i, j]
                    if np.isnan(v):
                        pair[i, j] = 0.0
                    elif np.isinf(v):
                        pair[i, j] = 1.7976931348623157e308 if v > 0 else -1.7976931348623157e308
            for i in range(2):
                for j in range(2):
                    trans[i, k + j] += pair[i, j]
            gamma[0, t] = pair[0, 0] + pair[1, 0]
            gamma[1, t] = pair[0, 1] + pair[1, 1]
            for n in range(num_subparts):
                d = alldata[n, s + t]
                if d:
                    c = 2 * n + (1 if d == 2 else 0)
                    emission[0, c] += gamma[0, t]
                    emission[1, c] += gamma[1, t]
                if f:
                    dp = alldata[n, s + T - 1]
                    if dp:
                        c = 2 * n + (1 if dp == 2 else 0)
                        emission[0, c] += gamma[0, T - 1]
                        emission[1, c] += gamma[1, T - 1]
            f = False
        init[0, 0] += gamma[0, 0]
        init[1, 0] += gamma[1, 0]
        for t in range(T):
            alpha_out[0, s + t] += alpha[0, t]
            alpha_out[1, s + t] += alpha[1, t]
    return trans, emission, init, loglike, alpha_out


def inner(x):
    """Drop-in for ``pyBKT.fit.EM_fit.inner``: same inputs, same outputs."""
    alldata = np.ascontiguousarray(x["alldata"])
    bigT = alldata.shape[1]
    trans, emission, init, loglike, alpha_out = _inner(
        np.ascontiguousarray(x["As"], dtype=np.float64), np.ascontiguousarray(x["Bn"], dtype=np.float64),
        np.ascontiguousarray(x["initial_distn"], dtype=np.float64),
        np.ascontiguousarray(x["allresources"], dtype=np.int64), np.ascontiguousarray(x["starts"], dtype=np.int64),
        np.ascontiguousarray(x["lengths"], dtype=np.int64), int(x["num_resources"]), int(x["num_subparts"]),
        alldata, int(x["sequence_idx_start"]), int(x["sequence_idx_end"]), bigT)
    # pyBKT adds each sequence's alphas into a zeroed (2, bigT) array; one block covering all of them is the same sum.
    return [trans, emission, init, loglike, [(0, bigT, alpha_out)]]
