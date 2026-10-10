"""Step 2. The judge learns alone. How hard does it still pull on a fake?"""
from extra import extra

print('STEP 2  A judge that learns alone (the forger never changes)')
print('  The judge sees 128 real points and 128 fakes per step and learns to tell them apart.')
print('  "yes" = the chance the judge gives a fake of being real.')
print('  "pull" = the length of the arrow that the forger\'s goal draws on one fake (the slope of the goal at the fake).')
print('   judge steps    yes to a fake    pull, goal "not caught"    pull, goal "called real"')
rows = extra()['alone']
for step, yes, a, b in rows:
    print(f'   {int(step):>8,}      {yes * 100:>8.2f} %        {a:>10.4f}                {b:>10.4f}')
step, yes, a, b = rows[-1]
print(f'  After {int(step):,} steps the judge is sure. The goal "not caught" pulls {b / a:,.0f} times weaker than "called real".')
print(f'  The strongest pull of "not caught" was {rows[:, 2].max():.4f}, at step {int(rows[rows[:, 2].argmax(), 0])}.')
