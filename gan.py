"""A forger and a judge. The forger turns random numbers into a fake; the judge says how real a thing looks.
They are trained against each other (Goodfellow et al. 2014, Algorithm 1)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import ring, digits

torch.set_num_threads(1)
NOISE = 32                                         # random numbers that go into the forger


def net(inputs, outputs, squash=False):
    """Two layers of 128 units. The forger for pictures ends in tanh, so every square stays between -1 and 1."""
    layers = [nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, outputs)]
    return nn.Sequential(*layers, nn.Tanh()) if squash else nn.Sequential(*layers)


def judge_loss(judge, real, fake, goal):
    if goal == 'score':                            # Wasserstein: real things get a high score, fakes a low one
        mixed = torch.rand(len(real), 1)
        mixed = (mixed * real + (1 - mixed) * fake).requires_grad_(True)
        slope = torch.autograd.grad(judge(mixed).sum(), mixed, create_graph=True)[0].norm(dim=1)
        return judge(fake).mean() - judge(real).mean() + 10 * ((slope - 1) ** 2).mean()
    ones, zeros = torch.ones(len(real), 1), torch.zeros(len(fake), 1)
    return F.binary_cross_entropy_with_logits(judge(real), ones) + F.binary_cross_entropy_with_logits(judge(fake), zeros)


def forger_loss(verdict, goal):
    """verdict = what the judge says about a batch of fakes (before the squash to 0..1)."""
    if goal == 'not_caught':                       # the paper's equation 1: make log(1 - D(G(z))) small
        return -F.binary_cross_entropy_with_logits(verdict, torch.zeros_like(verdict))
    if goal == 'called_real':                      # the paper's trick: make log D(G(z)) big
        return F.binary_cross_entropy_with_logits(verdict, torch.ones_like(verdict))
    return -verdict.mean()                         # 'score': make the judge's score of the fakes high


def train(what='ring', goal='not_caught', seed=0, steps=20000, lr=1e-4, judge_steps=1, batch=128, check=250, watch=None, noise=NOISE):
    """Play the game. `watch(step, forger, judge, loss_j, loss_f)` is called at every check."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pictures = torch.from_numpy(digits()[0]) if what == 'digits' else None
    size = 64 if what == 'digits' else 2
    forger, judge = net(noise, size, squash=what == 'digits'), net(size, 1)
    opt_f = torch.optim.Adam(forger.parameters(), lr, betas=(0.5, 0.999))
    opt_j = torch.optim.Adam(judge.parameters(), lr, betas=(0.5, 0.999))
    for step in range(1, steps + 1):
        for _ in range(judge_steps):               # the judge looks at real things and at fakes
            real = pictures[torch.from_numpy(rng.integers(0, len(pictures), batch))] if what == 'digits' else torch.from_numpy(ring(batch, rng))
            fake = forger(torch.randn(batch, noise)).detach()
            loss_j = judge_loss(judge, real, fake, goal)
            opt_j.zero_grad(); loss_j.backward(); opt_j.step()
        loss_f = forger_loss(judge(forger(torch.randn(batch, noise))), goal)   # the forger asks the judge
        opt_f.zero_grad(); loss_f.backward(); opt_f.step()
        if watch and step % check == 0:
            watch(step, forger, judge, loss_j.item(), loss_f.item())
    return forger, judge
