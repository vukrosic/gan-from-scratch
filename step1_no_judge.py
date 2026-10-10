"""Step 1. A forger without a judge. Its goal: be close to a real thing."""
import numpy as np
from data import digits, draw
from extra import extra
from measure import ring_score, digit_score, referee

e = extra()
p = e['nojudge_ring']
s = ring_score(p)
print('STEP 1  A forger without a judge')
print('The ring: 2,000 fakes after 5,000 steps of "be close to a real point".')
print(f'  on one of the eight spots: {s["on"] * 100:.0f} %   spots covered: {s["spots"]} of 8')
print(f'  the fakes are at ({p[:, 0].mean():+.2f}, {p[:, 1].mean():+.2f}), {np.linalg.norm(p, axis=1).mean():.2f} from the middle. The spots are 2.00 from the middle.')
print(f'  how far the fakes differ from each other: {p.std(0).mean():.3f}')
d = e['nojudge_digits']
s = digit_score(d[:1000])
print('The digits: the same goal on pictures.')
print(f'  the squares of two fakes differ by {d.std(0).mean():.3f} on average (real pictures: {digits()[0].std(0).mean():.2f})')
print(f'  the referee reads the digit {int(np.argmax(s["share"]))} in {max(s["share"]) * 100:.0f} % of them. Kinds of digit: {s["kinds"]} of 10.')
print('  one fake, and the average of all 1,797 real pictures:')
a, b = draw(d[0]).split('\n'), draw(digits()[0].mean(0)).split('\n')
for x, y in zip(a, b):
    print('   ', x, '   ', y)
print(f'  largest difference between the fake and the average picture: {np.abs(d[0] - digits()[0].mean(0)).max():.2f} (a square goes from -1 to 1)')
