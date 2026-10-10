"""Step 6. The judge gives a score, not a yes or no (Wasserstein GAN, 2017). And a new judge tests every forger."""
import numpy as np
from extra import extra
from runs import run, SEEDS, GOALS

WORD = {'not_caught': 'not caught (2014)   ', 'called_real': 'called real (advice)', 'score': 'score (2017)        '}

if __name__ == '__main__':
    print('STEP 6  A score in place of a yes or no')
    r = run('ring', 'score', 'big', 0)
    print('  The pull on a fake during run 0 on the ring (turn: pull):', '  '.join(f'{int(t):,}: {p:.2f}' for t, p in zip(r['step'][3::19], r['push'][3::19])))
    print('  All six versions, six runs each. "evenness" = the rarest kind and the most common kind, in % of the fakes.')
    print('   goal                   steps   ring: spots    digits: kinds        sure   evenness')
    for goal in GOALS:
        for size in ['small', 'big']:
            a = [run('ring', goal, size, s) for s in SEEDS]
            b = [run('digits', goal, size, s) for s in SEEDS]
            print(f'   {WORD[goal]}   {size:5}   ' + ' '.join(str(int(r['count'][-1])) for r in a) + '    ' + ' '.join(f'{int(r["count"][-1]):>2}' for r in b)
                  + f'   {np.mean([r["good"][-1] for r in b]) * 100:4.0f} %   {np.mean([r["share"][-1].min() for r in b]) * 100:4.1f} to {np.mean([r["share"][-1].max() for r in b]) * 100:4.1f}')
    e = extra()
    print('  A new judge learns from zero against each finished forger (big steps) and is tested on pictures it has not seen.')
    print('  Share it names correctly. A coin gives 50 %.')
    for goal, row in zip(GOALS, e['newjudge']):
        print(f'   {WORD[goal]}   ' + ' '.join(f'{v * 100:4.1f}' for v in row) + f'    mean {row.mean() * 100:.1f} %')
    print(f'   real pictures against other real pictures: {e["newjudge_real"] * 100:.1f} %')
