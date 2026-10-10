"""How good are the fakes? Counted, not judged by eye."""
import numpy as np
import torch
import torch.nn as nn
from data import SPOTS, SPREAD, digits
from gan import NOISE, net

FIXED = torch.randn(2000, NOISE, generator=torch.Generator().manual_seed(123))    # the same noise at every check


def ring_score(points):
    """Share of the fakes that land on each of the eight spots (within three spreads of its middle)."""
    d = np.linalg.norm(points[:, None, :] - SPOTS[None], axis=2)
    near, hit = d.argmin(1), d.min(1) < 3 * SPREAD
    share = np.array([(hit & (near == k)).mean() for k in range(8)])
    return {'spots': int((share >= 0.02).sum()), 'on': float(hit.mean()), 'share': share.round(4).tolist()}


_ref = {}
def referee():
    """A third network that only reads digits. It learns from 1,437 real pictures and is tested on the other 360."""
    if not _ref:
        x, y = digits()
        order = np.random.default_rng(0).permutation(len(x)); tr, te = order[:1437], order[1437:]
        torch.manual_seed(0)
        ref = net(64, 10); opt = torch.optim.Adam(ref.parameters(), 1e-3)
        xt, yt = torch.from_numpy(x[tr]), torch.from_numpy(y[tr])
        for _ in range(600):
            b = torch.randint(0, len(tr), (128,))
            loss = nn.functional.cross_entropy(ref(xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            _ref['right'] = float((ref(torch.from_numpy(x[te])).argmax(1).numpy() == y[te]).mean())
        _ref['net'] = ref
    return _ref['net'], _ref['right']


def digit_score(fakes):
    """Which digit does the referee read in each fake, and how close is the nearest real picture?"""
    ref, _ = referee()
    x = torch.as_tensor(fakes)
    with torch.no_grad():
        p = torch.softmax(ref(x), 1)
    share = np.bincount(p.argmax(1).numpy(), minlength=10) / len(x)
    near = torch.cdist(x, torch.from_numpy(digits()[0])).min(1).values
    return {'kinds': int((share >= 0.03).sum()), 'sure': float((p.max(1).values > 0.9).float().mean()),
            'near': float(near.mean()), 'share': share.round(4).tolist()}


def push(forger, judge, goal, n=512):
    """How hard does the forger's goal pull on one fake? Mean length of the pull, per fake."""
    from gan import forger_loss
    fake = forger(FIXED[:n, :forger[0].in_features]).detach().requires_grad_(True)
    g = torch.autograd.grad(forger_loss(judge(fake), goal) * n, fake)[0]
    return float(g.norm(dim=1).mean())
