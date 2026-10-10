"""Three small measurements outside the game, saved in files/extra.npz.
1. A forger trained without a judge: its goal is "be close to a real thing" (squared error).
2. A judge trained alone against a forger that never learns: how hard do the two goals still pull?
3. A new judge, trained from zero against a finished forger: can it tell the fakes from real things?
"""
import numpy as np
import torch
import torch.nn.functional as F
from data import ring, digits
from gan import NOISE, net, judge_loss
from measure import FIXED, push, ring_score, digit_score

FILE = __import__('pathlib').Path(__file__).parent / 'files' / 'extra.npz'


def no_judge(what, steps=5000):
    """Each fake is compared with a real thing drawn at random. The forger is told the distance, squared."""
    torch.manual_seed(0); rng = np.random.default_rng(0)
    pictures = torch.from_numpy(digits()[0])
    forger = net(NOISE, 64 if what == 'digits' else 2, squash=what == 'digits')
    opt = torch.optim.Adam(forger.parameters(), 1e-3)
    for _ in range(steps):
        real = pictures[torch.from_numpy(rng.integers(0, len(pictures), 128))] if what == 'digits' else torch.from_numpy(ring(128, rng))
        loss = F.mse_loss(forger(torch.randn(128, NOISE)), real)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return forger(FIXED).numpy()


def judge_alone(marks=(0, 25, 50, 100, 250, 500, 1000, 2000)):
    """The forger stays as it was born. Only the judge learns."""
    torch.manual_seed(0); rng = np.random.default_rng(0)
    forger, judge = net(NOISE, 2), net(2, 1)
    opt = torch.optim.Adam(judge.parameters(), 1e-4, betas=(0.5, 0.999))
    rows = []
    for step in range(marks[-1] + 1):
        if step in marks:
            with torch.no_grad():
                yes = torch.sigmoid(judge(forger(FIXED[:512]))).mean().item()
            rows.append([step, yes, push(forger, judge, 'not_caught'), push(forger, judge, 'called_real')])
        loss = judge_loss(judge, torch.from_numpy(ring(128, rng)), forger(torch.randn(128, NOISE)).detach(), 'not_caught')
        opt.zero_grad(); loss.backward(); opt.step()
    return np.array(rows, dtype=np.float32)


def new_judge(fakes, real, steps=3000, seed=0):
    """Half of the real things and half of the fakes are for learning, the other half for the test.
    Returns the share of the test things the new judge names correctly. 0.5 is a coin."""
    torch.manual_seed(seed)
    fakes, real = torch.as_tensor(fakes), torch.as_tensor(real)
    n = min(len(fakes), len(real)) // 2
    judge = net(fakes.shape[1], 1); opt = torch.optim.Adam(judge.parameters(), 1e-3)
    for _ in range(steps):
        a, b = torch.randint(0, n, (64,)), torch.randint(0, n, (64,))
        loss = judge_loss(judge, real[a], fakes[b], 'not_caught')
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        right = ((judge(real[n:2 * n]) > 0).float().mean() + (judge(fakes[n:2 * n]) < 0).float().mean()) / 2
    return float(right)


def real_halves():
    """The real pictures in a fixed random order: the first half for learning, the second for the test."""
    x = digits()[0]
    return x[np.random.default_rng(1).permutation(len(x))]


def all_new_judges():
    """A new judge against the finished forger of every picture run with big steps. Row = goal, column = run."""
    from runs import run, forger_of, GOALS, SEEDS
    real = real_halves()
    rows = []
    for goal in GOALS:
        rows.append([])
        for seed in SEEDS:
            with torch.no_grad():
                fakes = forger_of(run('digits', goal, 'big', seed), 'digits')(FIXED).numpy()
            rows[-1].append(new_judge(fakes, real))
    return np.array(rows, dtype=np.float32)


def extra():
    if not FILE.exists():
        out = {'alone': judge_alone(), 'nojudge_ring': no_judge('ring'), 'nojudge_digits': no_judge('digits'), 'newjudge': all_new_judges()}
        real = real_halves()
        out['newjudge_real'] = np.float32(new_judge(real[:898], real[898:]))   # real against real: nothing to find
        FILE.parent.mkdir(exist_ok=True)
        np.savez_compressed(FILE, **out)
    return dict(np.load(FILE))


if __name__ == '__main__':
    e = extra()
    print('new judge, share named correctly (0.5 is a coin). Rows: not caught, called real, score. Columns: six runs.')
    print(e['newjudge'].round(3)); print('real pictures against other real pictures:', e['newjudge_real'].round(3))
