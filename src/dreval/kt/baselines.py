"""The baselines: BKT (pyBKT), IRT (Rasch, ability updated online), PFA and DKT, plus each skill's mean.

Every model is fitted on the training students only and then predicts each test interaction from that student's
earlier interactions, before it sees the answer. An interaction with several skills gets the mean of its skills'
predictions (BKT, DKT) or one prediction from all of them (IRT, PFA).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from dreval.kt.data import Student


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def skill_mean(train: list[Student], test: list[Student]) -> list[np.ndarray]:
    right, total = defaultdict(float), defaultdict(float)
    for st in train:
        for it in st.items:
            for s in it.skills:
                right[s] += it.correct
                total[s] += 1
    overall = sum(right.values()) / sum(total.values())
    rate = {s: (right[s] + 10 * overall) / (total[s] + 10) for s in total}
    return [np.array([np.mean([rate.get(s, overall) for s in it.skills]) for it in st.items]) for st in test]


# -- BKT ---------------------------------------------------------------------


def _patch_pybkt(compiled: bool = True) -> None:
    """pyBKT 1.4.3 stores a (1, 1) array into one element of its log-likelihood array, which NumPy 2.5 refuses;
    hand it the scalar instead. With ``compiled`` its E-step runs as the numba copy in ``fast_bkt`` (same
    arithmetic, hours faster on EdNet). Nothing else of the fit changes."""
    from pyBKT.fit import EM_fit

    if not hasattr(EM_fit, "_python_inner"):
        EM_fit._python_inner = EM_fit.inner
    if compiled:
        from dreval.kt import fast_bkt

        EM_fit.inner = fast_bkt.inner
    else:
        EM_fit.inner = EM_fit._python_inner
    if getattr(EM_fit.run, "_scalar", False):
        return
    original = EM_fit.run

    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        result["total_loglike"] = float(np.asarray(result["total_loglike"]).reshape(-1)[0])
        return result

    run._scalar = True
    EM_fit.run = run


def bkt(train: list[Student], test: list[Student], seed: int = 0) -> list[np.ndarray]:
    """pyBKT's standard model, one per skill, fitted by EM on the training students' rows."""
    import pandas as pd
    from pyBKT.models import Model

    _patch_pybkt()

    def frame(students):
        rows = []
        for st in students:
            for number, it in enumerate(st.items):
                for s in it.skills:
                    rows.append((st.id, f"{st.id}:{number}", number, s, it.correct))
        return pd.DataFrame(rows, columns=["user_id", "key", "order_id", "skill_name", "correct"])

    fitted = Model(seed=seed, num_fits=5, parallel=False)
    train_df = frame(train)
    fitted.fit(data=train_df[["order_id", "user_id", "skill_name", "correct"]])
    test_df = frame(test)
    known = set(fitted.coef_)
    unseen = test_df[~test_df["skill_name"].isin(known)]
    seen = test_df[test_df["skill_name"].isin(known)]
    # predict() sorts its frame by user and order; read each row back by its own key, not by position.
    predicted = fitted.predict(data=seen[["order_id", "user_id", "skill_name", "correct", "key"]].copy())
    by_key = defaultdict(list)
    for key, p in zip(predicted["key"], predicted["correct_predictions"]):
        by_key[key].append(float(p))
    overall = float(train_df["correct"].mean())
    for key in unseen["key"]:
        by_key[key].append(overall)
    return [np.array([np.mean(by_key[f"{st.id}:{n}"]) for n in range(len(st.items))]) for st in test]


# -- IRT ---------------------------------------------------------------------


def _rasch(train: list[Student], l2: float = 1.0, steps: int = 200):
    """Student abilities and problem difficulties by L2-penalised maximum likelihood (Newton on the diagonal
    approximation would be slow to converge; L-BFGS on the full likelihood)."""
    from scipy.optimize import minimize

    problems = sorted({it.problem for st in train for it in st.items})
    p_index = {p: i for i, p in enumerate(problems)}
    s_idx, p_idx, y = [], [], []
    for n, st in enumerate(train):
        for it in st.items:
            s_idx.append(n)
            p_idx.append(p_index[it.problem])
            y.append(it.correct)
    s_idx, p_idx, y = np.array(s_idx), np.array(p_idx), np.array(y, dtype=float)
    ns, np_ = len(train), len(problems)

    def loss(w):
        theta, beta, mu = w[:ns], w[ns:ns + np_], w[-1]
        x = mu + theta[s_idx] - beta[p_idx]
        p = sigmoid(x)
        g = p - y
        nll = float(np.sum(np.logaddexp(0, x) - y * x))
        gt = np.bincount(s_idx, g, ns) + l2 * theta
        gb = -np.bincount(p_idx, g, np_) + l2 * beta
        return nll + 0.5 * l2 * (theta @ theta + beta @ beta), np.concatenate([gt, gb, [g.sum()]])

    w = minimize(loss, np.zeros(ns + np_ + 1), jac=True, method="L-BFGS-B", options={"maxiter": steps}).x
    theta, beta, mu = w[:ns], w[ns:ns + np_], w[-1]
    return {p: beta[i] for p, i in p_index.items()}, float(mu), float(np.var(theta))


def irt(train: list[Student], test: list[Student]) -> list[np.ndarray]:
    """Rasch: difficulties from the training students; a test student's ability starts at the population prior
    and is updated after each answer (Laplace approximation), so a prediction uses only earlier answers.
    A problem never seen in training takes the mean difficulty of its skills' training problems."""
    beta, mu, var = _rasch(train)
    by_skill = defaultdict(list)
    for st in train:
        for it in st.items:
            for s in it.skills:
                by_skill[s].append(beta[it.problem])
    skill_beta = {s: float(np.mean(v)) for s, v in by_skill.items()}
    out = []
    for st in test:
        m, v, preds = 0.0, var, []
        for it in st.items:
            b = beta.get(it.problem)
            if b is None:
                b = float(np.mean([skill_beta.get(s, 0.0) for s in it.skills]))
            preds.append(float(sigmoid((mu + m - b) / math.sqrt(1 + math.pi * v / 8))))
            m, v = _laplace(m, v, mu - b, it.correct)
        out.append(np.array(preds))
    return out


def _laplace(m: float, v: float, offset: float, y: int) -> tuple[float, float]:
    z = m
    for _ in range(10):
        p = 1.0 / (1.0 + math.exp(-(offset + z)))
        grad = (y - p) - (z - m) / v
        hess = -p * (1 - p) - 1 / v
        z -= grad / hess
    p = 1.0 / (1.0 + math.exp(-(offset + z)))
    return z, 1.0 / (p * (1 - p) + 1 / v)


# -- PFA ---------------------------------------------------------------------


def _pfa_features(st: Student, index: dict[str, int]):
    k = len(index)
    wins, fails = np.zeros(k), np.zeros(k)
    rows = []
    for it in st.items:
        x = np.zeros(3 * k)
        for s in it.skills:
            j = index.get(s)
            if j is None:
                continue
            x[j] = 1.0
            x[k + j] = wins[j]
            x[2 * k + j] = fails[j]
        rows.append(x)
        for s in it.skills:
            j = index.get(s)
            if j is not None:
                (wins if it.correct else fails)[j] += 1
    return rows


def pfa(train: list[Student], test: list[Student]) -> list[np.ndarray]:
    """Performance factors analysis: logit = Σ over the interaction's skills (β_k + γ_k·successes_k + ρ_k·failures_k),
    counts taken before the interaction (Pavlik et al. 2009). Counts enter as log(1 + n), a common variant that
    keeps long sequences from saturating."""
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression

    skills = sorted({s for st in train for it in st.items for s in it.skills})
    index = {s: i for i, s in enumerate(skills)}

    def matrix(students):
        rows = [r for st in students for r in _pfa_features(st, index)]
        x = np.array(rows)
        k = len(index)
        x[:, k:] = np.log1p(x[:, k:])
        return sparse.csr_matrix(x)

    y = np.array([it.correct for st in train for it in st.items])
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(matrix(train), y)
    out, flat = [], model.predict_proba(matrix(test))[:, 1]
    at = 0
    for st in test:
        out.append(flat[at:at + len(st.items)])
        at += len(st.items)
    return out


# -- DKT ---------------------------------------------------------------------


def dkt(train: list[Student], test: list[Student], seed: int = 0, hidden: int = 100, epochs: int = 30,
        patience: int = 3) -> list[np.ndarray]:
    """Deep knowledge tracing (Piech et al. 2015): an LSTM reads (skills, correct) as a multi-hot input of 2K
    and outputs each skill's probability for the next interaction. 10% of the training students are held out
    for early stopping on their AUC."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(int(__import__("os").environ.get("DREVAL_TORCH_THREADS", "2")))
    rng = np.random.default_rng(seed)
    skills = sorted({s for st in train for it in st.items for s in it.skills})
    index = {s: i for i, s in enumerate(skills)}
    k = len(index)
    order = rng.permutation(len(train))
    hold = {int(i) for i in order[: len(train) // 10]}
    fit_set = [st for i, st in enumerate(train) if i not in hold]
    val_set = [st for i, st in enumerate(train) if i in hold]

    def tensors(st):
        n = len(st.items)
        x = torch.zeros(n, 2 * k)
        target = torch.zeros(n, k)
        y = torch.zeros(n)
        for t, it in enumerate(st.items):
            js = [index[s] for s in it.skills if s in index]
            for j in js:
                x[t, j + (k if it.correct else 0)] = 1.0
                target[t, j] = 1.0 / len(js)
            y[t] = it.correct
        return x, target, y

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(2 * k, hidden, batch_first=True)
            self.out = nn.Linear(hidden, k)
            self.start = nn.Parameter(torch.zeros(k))

        def forward(self, x):
            h, _ = self.lstm(x)
            logits = self.out(h)
            # The prediction for step t comes from steps before t; step 0 has a learned prior.
            return torch.cat([self.start.expand(x.shape[0], 1, k), logits[:, :-1]], dim=1)

    def batches(students, size, shuffle):
        idx = rng.permutation(len(students)) if shuffle else np.arange(len(students))
        for i in range(0, len(idx), size):
            chunk = [tensors(students[j]) for j in idx[i:i + size]]
            lengths = [c[0].shape[0] for c in chunk]
            pad = lambda parts: nn.utils.rnn.pad_sequence(parts, batch_first=True)
            mask = pad([torch.ones(n) for n in lengths])
            yield pad([c[0] for c in chunk]), pad([c[1] for c in chunk]), pad([c[2] for c in chunk]), mask, lengths

    def predict(net, students):
        net.eval()
        out = []
        with torch.no_grad():
            for x, target, _, _, lengths in batches(students, 64, False):
                logit = (net(x) * target).sum(-1) / target.sum(-1).clamp(min=1e-9)
                p = torch.sigmoid(logit)
                out += [p[i, :n].numpy().astype(float) for i, n in enumerate(lengths)]
        return out

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, waited = -1.0, None, 0
    for _ in range(epochs):
        net.train()
        for x, target, y, mask, _ in batches(fit_set, 32, True):
            logit = (net(x) * target).sum(-1)
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        val = predict(net, val_set)
        auc = roc_auc_score(np.concatenate([[it.correct for it in st.items] for st in val_set]), np.concatenate(val))
        if auc > best:
            best, best_state, waited = auc, {n: t.clone() for n, t in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    net.load_state_dict(best_state)
    return predict(net, test)
