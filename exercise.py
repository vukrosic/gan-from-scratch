"""Exercise. Change one number.

In step 4 the forger had the goal "be called real" and small steps. All of its fakes were one clump
that ran around the ring: never more than 1 spot of 8.
The forger gets 32 random numbers for every fake. Give it 2.

Guess first: more spots, fewer spots or the same? Then run this file (about 15 seconds).
"""
import time
import torch
from gan import train
from measure import FIXED, ring_score
from runs import run

t0 = time.time()
print('The goal "be called real", small steps, 2 random numbers in place of 32. Run 0:')


def watch(step, forger, judge, loss_j, loss_f):
    if step % 2000 == 0:
        with torch.no_grad():
            s = ring_score(forger(FIXED[:, :2]).numpy())
        print(f'  turn {step:>6,}: {s["spots"]} of 8 spots, {s["on"] * 100:3.0f} % of the fakes on a spot   ({time.time() - t0:.0f} s)', flush=True)


train('ring', 'called_real', seed=0, noise=2, watch=watch)
print('Three saved runs, spots at the end:')
print('  32 random numbers:', ' '.join(str(int(run('ring', 'called_real', 'small', s)['count'][-1])) for s in range(3)))
print('   2 random numbers:', ' '.join(str(int(run('ring', 'called_real', 'small', s, 'two')['count'][-1])) for s in range(3)))
print('Why could that be? A point on the page has 2 numbers. With 2 numbers in, the forger only has to bend the page.')
print('With 32 numbers in, it has to press 32 directions flat first, and one clump is the easiest way to do that.')
print('This is a guess, not a proof. Next exercise: try 4, 8 and 16 (the word noise in train) and find where the clump comes back.')
