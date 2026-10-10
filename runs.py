"""Every learning run of the video, saved in files/. A run that is already there is read, not made again.
python3 runs.py          makes every run that is missing (6 at a time)
"""
import sys
from pathlib import Path
import numpy as np
import torch

FILES = Path(__file__).parent / 'files'
SEEDS = range(6)
GOALS = ['not_caught', 'called_real', 'score']
STEP = {'small': 1e-4, 'big': 1e-3}
LIST = [(what, goal, size, seed) for what in ['ring', 'digits'] for goal in GOALS for size in STEP for seed in SEEDS]
# three more kinds of run, marked with a last word:
#   'five'  the judge takes five steps for every step of the forger
#   'two'   the forger gets 2 random numbers, not 32 (the exercise)
#   'fine'  the same run as without the word, looked at every 10 steps (for the pictures in the video)
LIST += [('ring', 'called_real', 'small', seed, word) for word in ['five', 'two'] for seed in range(3)]
LIST += [('ring', 'not_caught', 'small', 3, 'fine'), ('ring', 'called_real', 'small', 0, 'fine')]
GRID = np.stack(np.meshgrid(np.linspace(-3, 3, 24), np.linspace(-3, 3, 24)), -1).reshape(-1, 2).astype(np.float32)


def name(what, goal, size, seed, word=''):
    return FILES / f'{what}-{goal}-{size}-{seed}{"-" + word if word else ""}.npz'


def make(what, goal, size, seed, word=''):
    from gan import train, NOISE
    from measure import FIXED, ring_score, digit_score, push
    noise = 2 if word == 'two' else NOISE
    out = {k: [] for k in ['step', 'count', 'good', 'share', 'loss_j', 'loss_f', 'yes_real', 'yes_fake', 'push', 'shots', 'field', 'near']}
    real = torch.from_numpy(__import__('data').ring(512, np.random.default_rng(99))) if what == 'ring' else torch.from_numpy(__import__('data').digits()[0][:512])

    def watch(step, forger, judge, loss_j, loss_f):
        with torch.no_grad():
            fakes = forger(FIXED[:, :noise]).numpy()
            vr, vf = judge(real), judge(torch.from_numpy(fakes[:512]))
        s = ring_score(fakes) if what == 'ring' else digit_score(fakes[:1000])
        out['step'].append(step); out['loss_j'].append(loss_j); out['loss_f'].append(loss_f)
        out['count'].append(s['spots'] if what == 'ring' else s['kinds']); out['good'].append(s['on'] if what == 'ring' else s['sure'])
        out['share'].append(s['share']); out['near'].append(0.0 if what == 'ring' else s['near'])
        if goal == 'score':                       # a score has no "yes": count how often real beats the middle score
            mid = (vr.mean() + vf.mean()) / 2
            out['yes_real'].append(float((vr > mid).float().mean())); out['yes_fake'].append(float((vf > mid).float().mean()))
        else:
            out['yes_real'].append(float((vr > 0).float().mean())); out['yes_fake'].append(float((vf > 0).float().mean()))
        out['push'].append(push(forger, judge, goal))
        out['shots'].append(fakes[:128] if what == 'ring' else fakes[:16])
        if what == 'ring':
            with torch.no_grad():
                v = judge(torch.from_numpy(GRID)).squeeze(1)
            out['field'].append((v if goal == 'score' else torch.sigmoid(v)).numpy())

    forger, _ = train(what, goal, seed, lr=STEP[size], judge_steps=5 if goal == 'score' or word == 'five' else 1,
                      check=10 if word == 'fine' else 250, noise=noise, watch=watch)
    arrays = {k: np.array(v, dtype=np.float16 if k in ('shots', 'field') else np.float32) for k, v in out.items() if len(v)}
    arrays.update({f'w{i}': p.detach().numpy() for i, p in enumerate(forger.parameters())})
    FILES.mkdir(exist_ok=True)
    np.savez_compressed(name(what, goal, size, seed, word), **arrays)


def run(what, goal, size='small', seed=0, word=''):
    f = name(what, goal, size, seed, word)
    if not f.exists():
        make(what, goal, size, seed, word)
    return dict(np.load(f))


def forger_of(r, what):
    """The forger of a saved run, as it was after the last step."""
    from gan import net
    f = net(r['w0'].shape[1], 64 if what == 'digits' else 2, squash=what == 'digits')
    with torch.no_grad():
        for i, p in enumerate(f.parameters()):
            p.copy_(torch.from_numpy(r[f'w{i}']))
    return f


def _one(job):
    make(*job)
    return job


if __name__ == '__main__':
    from multiprocessing import Pool
    todo = [j for j in LIST if not name(*j).exists()]
    if len(sys.argv) > 1:
        todo = [j for j in todo if sys.argv[1] in '-'.join(map(str, j))]
    print(len(todo), 'runs to make', flush=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as pool:
        for k, job in enumerate(pool.imap_unordered(_one, todo)):
            print(k + 1, *job, flush=True)
