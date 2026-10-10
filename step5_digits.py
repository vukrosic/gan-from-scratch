"""Step 5. Pictures: 8 x 8 squares with a hand-written digit. A referee counts what the forger draws."""
import numpy as np
import torch
from data import digits, draw
from measure import referee, digit_score, FIXED
from runs import run, forger_of, SEEDS


def real_numbers():
    """The same counts for real pictures: the 360 the referee never saw."""
    x, y = digits()
    test = np.random.default_rng(0).permutation(len(x))[1437:]
    s = digit_score(x[test])
    d = torch.cdist(torch.from_numpy(x), torch.from_numpy(x))
    d.fill_diagonal_(99)
    return s, float(d.min(1).values.mean())


if __name__ == '__main__':
    print('STEP 5  The digits')
    s, near = real_numbers()
    print(f'  The referee learned from 1,437 real pictures and reads {referee()[1] * 100:.1f} % of the other 360 correctly.')
    print(f'  Real pictures: {s["kinds"]} kinds of digit, the referee is sure (over 90 %) for {s["sure"] * 100:.0f} %.')
    print(f'  A real picture and the most similar other real picture differ by {near:.2f}.')
    print('  A kind of digit counts when the referee reads it in 3 % of 1,000 fakes. Six runs each, 20,000 turns.')
    print('   goal          steps   kinds of digit       referee sure   nearest real picture')
    for goal, word in [('not_caught', 'not caught '), ('called_real', 'called real')]:
        for size in ['small', 'big']:
            rs = [run('digits', goal, size, seed) for seed in SEEDS]
            print(f'   {word}   {size:5}   ' + ' '.join(f'{int(r["count"][-1]):>2}' for r in rs) + f'      {np.mean([r["good"][-1] for r in rs]) * 100:4.0f} %           {np.mean([r["near"][-1] for r in rs]):.2f}')
    r = run('digits', 'called_real', 'big', 0)
    print('  Goal "called real", big steps, run 0, kinds of digit over the run:')
    print('   turn  ', ' '.join(f'{int(t):>6,}' for t in r['step'][3::8]))
    print('   kinds ', ' '.join(f'{int(c):>6}' for c in r['count'][3::8]))
    print('   sure  ', ' '.join(f'{g * 100:>5.0f}%' for g in r['good'][3::8]))
    print('   share of each digit 0 to 9 at turn 3,000: ', ' '.join(f'{v * 100:2.0f}' for v in r['share'][11]))
    print('   share of each digit 0 to 9 at turn 20,000:', ' '.join(f'{v * 100:2.0f}' for v in r['share'][-1]))
    print('   digits that are gone:', ' '.join(str(k) for k in range(10) if r['share'][-1][k] < 0.03))
    print('  Four fakes of the goal "not caught", big steps, run 0:')
    with torch.no_grad():
        f = forger_of(run('digits', 'not_caught', 'big', 0), 'digits')(FIXED[:4]).numpy()
    for line in zip(*[draw(p).split('\n') for p in f]):
        print('    ' + '    '.join(line))
