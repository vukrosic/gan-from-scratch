"""Step 4. The advice of the paper: the forger's goal becomes "be called real"."""
import numpy as np
from runs import run, SEEDS


def laps(r):
    """How often the middle of the fakes goes around the ring."""
    c = r['shots'].astype(np.float32).mean(1)
    turned = np.unwrap(np.arctan2(c[:, 1], c[:, 0]))
    return abs(turned[-1] - turned[0]) / (2 * np.pi)


if __name__ == '__main__':
    print('STEP 4  The same game, the forger\'s goal is now "be called real" (section 3 of the paper)')
    print('   run   spots at the end   most spots at any check   judge says yes to a fake')
    for seed in SEEDS:
        r = run('ring', 'called_real', 'small', seed)
        print(f'   {seed}         {int(r["count"][-1])} of 8                {int(r["count"].max())}                      {r["yes_fake"][-20:].mean() * 100:4.1f} %')
    r = run('ring', 'called_real', 'small', 0, 'fine')
    pts = r['shots'].astype(np.float32)
    late = r['step'] > 2200
    print(f'  Run 0, looked at every 10 turns: the fakes are one clump, {pts[late].std(1).mean():.2f} wide, {np.linalg.norm(pts[late].mean(1), axis=1).mean():.2f} from the middle.')
    print(f'  The clump goes around the ring {laps(r):.0f} times in 20,000 turns.')
    print(f'  Checks with two spots or more: {int((r["count"] >= 2).sum())} of {len(r["count"]):,}.')
    best = np.where(r['share'].max(1) > 0.2, r['share'].argmax(1), -1)
    order = [int(b) for i, b in enumerate(best) if b >= 0 and (i == 0 or best[i - 1] != b)]
    print('  Spots it passes, in order (first 24):', ' '.join(map(str, order[:24])))
    print('  The judge takes five steps per turn:')
    for seed in range(3):
        r = run('ring', 'called_real', 'small', seed, 'five')
        print(f'   run {seed}: {int(r["count"][-1])} spot, {r["good"][-20:].mean() * 100:.0f} % of the fakes on it, always spot {int(r["share"][-1].argmax())}; judge says yes to a fake {r["yes_fake"][-20:].mean() * 100:.0f} %, to a real point {r["yes_real"][-20:].mean() * 100:.0f} %')
    print('  Ten times bigger steps for both networks:')
    for goal, word in [('not_caught', 'not caught '), ('called_real', 'called real')]:
        rs = [run('ring', goal, 'big', s) for s in SEEDS]
        print(f'   goal "{word}": spots', ' '.join(str(int(r['count'][-1])) for r in rs), f'  on a spot {np.mean([r["good"][-1] for r in rs]) * 100:.0f} %')
