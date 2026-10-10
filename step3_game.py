"""Step 3. The game: one step for the judge, one step for the forger. The forger's goal is "not caught"."""
import numpy as np
from runs import run, SEEDS

print('STEP 3  The game with the goal of the 2014 paper (equation 1)')
print('  20,000 turns. A spot counts when 2 % of 2,000 fakes land on it.')
print('   run   spots covered   fakes on a spot   judge says yes to: real   fake')
for seed in SEEDS:
    r = run('ring', 'not_caught', 'small', seed)
    print(f'   {seed}        {int(r["count"][-1])} of 8          {r["good"][-1] * 100:4.0f} %                   {r["yes_real"][-20:].mean() * 100:4.0f} %  {r["yes_fake"][-20:].mean() * 100:4.0f} %')
r = run('ring', 'not_caught', 'small', 3, 'fine')
pts = r['shots'].astype(np.float32)
print('  Run 3, looked at every 10 turns:')
print('   turn     fakes are this far from the middle   spots   on a spot')
for turn in [10, 50, 100, 250, 1000, 3000, 6250, 9250, 12000, 20000]:
    i = turn // 10 - 1
    print(f'   {turn:>6,}            {np.linalg.norm(pts[i], axis=1).mean():.2f}                      {int(r["count"][i])}       {r["good"][i] * 100:4.1f} %')
first = {n: int(r['step'][np.argmax(r['count'] >= n)]) for n in (1, 4, 8)}
print(f'  First spot at turn {first[1]:,}, four spots at turn {first[4]:,}, all eight at turn {first[8]:,}.')
