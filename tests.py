"""Checks of the claims of the video. python3 tests.py"""
import numpy as np
import torch
import torch.nn.functional as F
from data import ring, digits, SPOTS
from extra import extra
from gan import NOISE, train, forger_loss
from measure import FIXED, ring_score, referee
from runs import run, forger_of, SEEDS, GOALS
from step4_advice import laps
from step5_digits import real_numbers

done = []


def check(text, ok):
    done.append(ok)
    print(('ok    ' if ok else 'WRONG ') + text)


six = lambda what, goal, size: [run(what, goal, size, s) for s in SEEDS]
end = lambda rs: [int(r['count'][-1]) for r in rs]
e = extra()

# the data
real = ring(2000, np.random.default_rng(5))
check('the ring has eight spots, each 2 from the middle', len(SPOTS) == 8 and np.allclose(np.linalg.norm(SPOTS, axis=1), 2))
check('real points cover 8 spots and 98 to 100 % of them count as on a spot', ring_score(real)['spots'] == 8 and 0.98 < ring_score(real)['on'] <= 1)
x, y = digits()
check('1,797 pictures of 64 squares, every square between -1 and 1', x.shape == (1797, 64) and x.min() == -1 and x.max() == 1)
check('the referee reads more than 98 % of the 360 pictures it never saw', referee()[1] > 0.98)

# the two goals
v = torch.linspace(-6, 6, 25).reshape(-1, 1)
d = torch.sigmoid(v)
check('goal "not caught" is the mean of log(1 - D)', torch.isclose(forger_loss(v, 'not_caught'), torch.log(1 - d).mean(), atol=1e-5).item())
check('goal "called real" is minus the mean of log D', torch.isclose(forger_loss(v, 'called_real'), -torch.log(d).mean(), atol=1e-5).item())
v.requires_grad_(True)
a = torch.autograd.grad(forger_loss(v, 'not_caught') * len(v), v)[0]
b = torch.autograd.grad(forger_loss(v, 'called_real') * len(v), v)[0]
check('the pull of "not caught" is D, the pull of "called real" is 1 - D', torch.allclose(-a, d, atol=1e-5) and torch.allclose(-b, 1 - d, atol=1e-5))

# the same seed gives the same run
def short(seed):
    f, _ = train('ring', 'called_real', seed, steps=200)
    with torch.no_grad():
        return f(FIXED[:8]).numpy()
check('a run with the same seed gives the same fakes again', np.array_equal(short(0), short(0)) and not np.array_equal(short(0), short(1)))
r = run('ring', 'called_real', 'small', 0)
with torch.no_grad():
    again = forger_of(r, 'ring')(FIXED[:128]).numpy()
check('a saved forger draws its saved fakes again', np.allclose(again, r['shots'][-1].astype(np.float32), atol=2e-3))
fine = run('ring', 'called_real', 'small', 0, 'fine')
check('looking every 10 turns does not change the run', np.allclose(fine['good'][24::25], r['good']))

# step 1 and 2
p = e['nojudge_ring']
check('without a judge: 0 spots, every fake less than 0.1 from the middle', ring_score(p)['spots'] == 0 and np.linalg.norm(p, axis=1).max() < 0.1)
check('without a judge the picture is the average of all real pictures', np.abs(e['nojudge_digits'] - x.mean(0)).max() < 0.1)
al = e['alone']
check('a judge that learned alone for 2,000 steps says yes to a fake in under 1 %', al[-1, 1] < 0.01)
check('then "not caught" pulls more than 500 times weaker than "called real"', al[-1, 3] / al[-1, 2] > 500)

# step 3 and 4
check('goal "not caught", small steps: 7 or 8 spots in all six runs', min(end(six('ring', 'not_caught', 'small'))) >= 7)
check('goal "called real", small steps: never more than 1 spot, at any check of any run', max(r['count'].max() for r in six('ring', 'called_real', 'small')) <= 1)
check('the clump goes around the ring more than 30 times', laps(fine) > 30)
five = [run('ring', 'called_real', 'small', s, 'five') for s in range(3)]
check('five judge steps per turn: one spot with over 97 % of the fakes, the same spot in the last 20 checks',
      all(r['count'][-20:].max() == 1 and r['good'][-20:].mean() > 0.97 and len(set(r['share'][-20:].argmax(1))) == 1 for r in five))
check('big steps: "not caught" 8 spots in all runs, "called real" loses a spot in some', end(six('ring', 'not_caught', 'big')) == [8] * 6 and min(end(six('ring', 'called_real', 'big'))) < 8)

# step 5
s, near = real_numbers()
adv = six('digits', 'called_real', 'big')
check('real pictures: 10 kinds, the referee is sure for over 90 %', s['kinds'] == 10 and s['sure'] > 0.9)
check('"called real", big steps: every run had all 10 kinds once and ends with 6 or 7', all(r['count'].max() == 10 for r in adv) and set(end(adv)) <= {6, 7})
check('its fakes are nearer to a real picture than real pictures are to each other', max(r['near'][-1] for r in adv) < near)
check('"not caught": 10 kinds in all twelve runs', end(six('digits', 'not_caught', 'small')) + end(six('digits', 'not_caught', 'big')) == [10] * 12)
check('its fakes with big steps are not nearer to a real picture than real pictures are to each other', min(r['near'][-1] for r in six('digits', 'not_caught', 'big')) > near)

# step 6
check('the score: 8 spots and 10 kinds in all 24 runs', all(end(six('ring', 'score', z)) == [8] * 6 and end(six('digits', 'score', z)) == [10] * 6 for z in ['small', 'big']))
pull = np.concatenate([r['push'][3:] for r in six('ring', 'score', 'big')])
check('with the score the pull on a fake stays between 0.9 and 1.1', pull.min() > 0.9 and pull.max() < 1.1)
nj = e['newjudge'].mean(1)
check('a new judge names the fakes of the score least often, those of "called real" most often', nj[2] < nj[0] < nj[1])
check('a new judge still beats a coin against every forger', e['newjudge'].min() > 0.55)
check('a new judge cannot tell real pictures from other real pictures', abs(float(e['newjudge_real']) - 0.5) < 0.05)

# the exercise
check('2 random numbers in place of 32: 8 spots in three runs', [int(run('ring', 'called_real', 'small', s, 'two')['count'][-1]) for s in range(3)] == [8, 8, 8])

print('ALL TESTS PASS' if all(done) else 'SOME TESTS ARE WRONG', f'({sum(done)} of {len(done)})')
