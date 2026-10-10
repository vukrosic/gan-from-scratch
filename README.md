# GAN from scratch

Code for the video **Two Networks That Fight Until One Can Draw**.
Two small neural networks learn against each other. The forger (the paper's generator) turns random numbers into a fake. The judge (the discriminator) gets real things and fakes and says which is which. The forger never sees a real thing. All it gets is the direction in which each fake would look more real to the judge.
Six small programs build the method of Goodfellow et al. (2014) from an empty file and measure three goals for the forger: the goal of the paper's equation 1, the goal the paper advises in its place, and the score of 2017 (Wasserstein GAN with gradient penalty). About 610 lines of Python 3 with numpy, torch and scikit-learn.

```bash
pip install numpy torch scikit-learn
sh run_all.sh         # six steps, the tests and the exercise; expected output is in demo-output.txt
```

Every learning run is saved in `files/` (80 runs, 22 MB), so the steps only read the runs. All of `run_all.sh` takes about half a minute; the exercise trains one pair of networks and takes about 15 seconds of it.
To measure everything again, delete `files/` and run `python3 runs.py` (6 runs at a time). One run of 20,000 turns takes about 15 seconds on a laptop, and about 70 seconds with the score. A run with the same seed gives the same fakes again (`tests.py` checks this).

Two data sets, both on your disk after the install, nothing to download:

- **The ring.** Points on a page. The real points sit in eight spots on a ring of radius 2. A spot counts as covered when 2 % of 2,000 fakes land on it.
- **The digits.** The 1,797 handwritten digits of scikit-learn, 8 squares by 8. A third network, the referee, is trained on 1,437 of them with their names and reads 98.6 % of the other 360 correctly. A kind of digit counts when the referee reads it in 3 % of 1,000 fakes.

| File | Idea | What the run shows |
| --- | --- | --- |
| `data.py` | The ring and the digits | 25 lines. |
| `gan.py` | The two networks, the three goals, the game | 57 lines. `train()` plays 20,000 turns: the judge takes one step with 128 real things and 128 fakes, then the forger takes one step. |
| `step1_no_judge.py` | A forger without a judge | The goal "be close to a real point drawn at random" puts all 2,000 fakes 0.01 from the middle of the ring, 0 of 8 spots. On the digits it draws one grey picture, the average of all 1,797 real pictures. |
| `step2_judge.py` | A judge that learns alone | After 2,000 steps the judge gives a fake a chance of 0.13 % of being real. The goal "not caught" then pulls on a fake with 0.0034, the goal "called real" with 2.37: 700 times stronger. |
| `step3_game.py` | The game with the goal of the paper | Six runs: 8, 7, 8, 8, 8 and 7 spots of 8, with 76 to 89 % of the fakes on a spot. Run 3: four spots at turn 6,100, all eight at turn 8,920. |
| `step4_advice.py` | The goal the paper advises | Five of six runs end on 1 spot, one on 0. No check of any run has two spots. The fakes are one clump that goes around the ring 33 times. With five judge steps per turn the clump stops on one spot with 98 % of the fakes. With ten times bigger steps: 8, 7, 6, 7, 7 and 8 spots. |
| `step5_digits.py` | The same test on pictures | "Not caught": 10 kinds of digit in all twelve runs. "Called real" with big steps: every run had 10 kinds once and ends with 6 or 7; the referee is sure about 93 % of its fakes (real pictures: 95 %). |
| `step6_score.py` | A score in place of a chance | 8 spots and 10 kinds in all 24 runs. The pull on a fake stays near 1. A new judge, trained against each finished forger, is right in 85 % of new pictures against "called real", 79 % against "not caught", 62 % against the score, and 51 % for real against real. |
| `measure.py` | Spots, kinds, the referee, the pull | 54 lines. |
| `runs.py` | The saved runs | Makes every run that is missing in `files/` and reads the ones that exist. |
| `extra.py` | The forger without a judge, the judge alone, the new judges | Saved in `files/extra.npz`. |
| `tests.py` | Checks of the claims | 30 tests. |
| `exercise.py` | Change one number | See below. |

## Reading the numbers

- **Six runs per version are few**, and all of them use one pair of small networks (two hidden layers of 128), one batch size and the Adam optimizer.
- **Two step sizes.** "Small" is 0.0001 and "big" is 0.001 for both networks. The advice collapses on the ring with small steps and finds 6 to 8 spots with big steps. On the digits it is the other way around: 9 or 10 kinds with small steps, 6 or 7 with big steps. The video shows the digits with big steps and says so.
- **The warning of the paper did not bite here.** The paper advises "called real" because "not caught" gives almost no pull when the judge is far ahead. That is true against a judge trained alone (step 2). In the real games the pull of "not caught" was never below 0.078 in twelve runs, because the judge never got that far ahead.
- **Sharp is not the same as varied.** The fakes of "called real" are the ones the referee is most sure about, and they sit nearer to a real picture (1.81) than real pictures sit to each other (2.05). They also miss three or four digits.
- **"Spots covered" forgives a lot.** A spot counts from 2 % of the fakes. The last column of step 3 says how many fakes are on any spot at all.

## What this is not

- **Not a reproduction of the papers.** The 2014 paper trains on MNIST, faces and CIFAR-10. This is a ring of points and 8 × 8 digits.
- **Not a ranking of GAN goals.** Lucic et al. (2018) compared many goals with a large search of settings and found them close together. The order here holds for this setup.
- **No picture-quality score.** There is no FID here. Variety is counted with a referee, and realism with a new judge.
- **The score costs time.** Its judge takes five steps per turn and each step works out one more slope: about 3.2 ms per turn against 0.7 ms.

Sources: Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014, arXiv 1406.2661. Arjovsky, Chintala, Bottou, "Wasserstein GAN", arXiv 1701.07875 (2017). Gulrajani et al., "Improved Training of Wasserstein GANs", arXiv 1704.00028 (2017). Arjovsky and Bottou, "Towards Principled Methods for Training Generative Adversarial Networks", arXiv 1701.04862 (2017). Lucic et al., "Are GANs Created Equal? A Large-Scale Study", arXiv 1711.10337 (2018).

## Try it

In step 4 the forger had the goal "be called real" and small steps, and its fakes were one clump: never more than 1 spot of 8. The forger gets 32 random numbers for every fake. `exercise.py` gives it 2.

```python
train('ring', 'called_real', seed=0, noise=2, watch=watch)
```

Guess first: more spots, fewer spots or the same? Then run `python3 exercise.py` (about 15 seconds). It prints the answer, one possible reason and a second exercise: try 4, 8 and 16 random numbers and find where the clump comes back.

MIT licence.
