# aml_experiment.py
# ------------------------------------------------------------
# Validate that Atomized‑Semilattice Learning (AML) fits
# every trace‑consistent data set with zero training error.
#
# Requires:  PyTorch ≥ 1.9
# ------------------------------------------------------------

import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim


# ----------  AML model  -----------------------------------------------------
class AML(nn.Module):
    """
    Each 'atom' i gets a learned scalar ϕ_i.
    The score (trace) of a set S is max_{i∈S} ϕ_i.
    """
    def __init__(self, num_atoms: int):
        super().__init__()
        # Start with small random values so the margin loss has room to improve
        self.phi = nn.Parameter(0.01 * torch.randn(num_atoms))

    def forward(self, batch_sets):
        # batch_sets is a list[list[int]] of atom indices
        traces = [self.phi[idxs].max() if idxs else self.phi.min()
                  for idxs in batch_sets]
        return torch.stack(traces)  # shape = (batch,)


# ----------  Loss function  -------------------------------------------------
def trace_hinge_loss(pos_scores: torch.Tensor,
                     neg_scores: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
    """
    For every positive P and negative N we want:  trace(P) ≥ trace(N) + margin.
    """
    diff = margin - (pos_scores[:, None] - neg_scores[None, :])
    return diff.clamp(min=0).mean()


# ----------  Synthetic data generators  ------------------------------------
def synth_consistent(num_atoms=20, num_pos=10, num_neg=10):
    """
    Create a *trace‑consistent* data set.
    Pick a hidden 'true' sup‑set U; all positives intersect U, negatives avoid U.
    """
    U = set(random.sample(range(num_atoms), k=random.randint(4, num_atoms - 4))) # Ensure at least 4 elements are left for complement
    pos_sets, neg_sets = [], []

    for _ in range(num_pos):
        # Convert the set U to a list before using it in random.sample
        extra = set(random.sample(list(U), k=random.randint(1, len(U))))
        pos_sets.append(list(U | extra))

    complement = set(range(num_atoms)) - U
    for _ in range(num_neg):
        # Ensure complement has more than one element to sample from
        if len(complement) > 1:
            neg = set(random.sample(list(complement), k=random.randint(1, len(complement))))
            neg_sets.append(list(neg))
        else:
            # Handle the case when complement is empty or has one element.
            # You can either skip the iteration or add an empty set to neg_sets
            # based on the desired behavior.
            # Here, we add an empty set:
            neg_sets.append([])
    return pos_sets, neg_sets


def synth_inconsistent(num_atoms=20):
    """
    Create a *trace‑inconsistent* data set by making one negative a subset of U.
    """
    U = set(random.sample(range(num_atoms), k=10))
    pos_sets = [list(U)]
    neg_sets = [list(set(random.sample(U, k=3)))]  # ⊆ U  ⇒ impossible
    # Add a few more random neg/pos sets so it isn't trivial
    pos_sets += [list(U | {random.randrange(num_atoms)})]
    neg_sets += [list({random.randrange(num_atoms)})]
    return pos_sets, neg_sets


# ----------  Training loop per trial  --------------------------------------
def run_trial(pos_sets, neg_sets, num_atoms, epochs, lr, verbose=False):
    device = torch.device("cpu")
    model = AML(num_atoms).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)

    all_sets = pos_sets + neg_sets          # list[list[int]]
    labels   = torch.tensor([1]*len(pos_sets) + [0]*len(neg_sets))

    for epoch in range(epochs):
        opt.zero_grad()
        scores = model(all_sets)
        loss = trace_hinge_loss(scores[labels == 1],
                                scores[labels == 0])
        loss.backward()
        opt.step()

        if verbose and (epoch % 20 == 0 or epoch == epochs-1):
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}")

    # Convert scores to 1/0 predictions by sign (threshold 0)
    preds = (scores.detach() > 0).int()
    accuracy = (preds == labels).float().mean().item()
    return accuracy, loss.item()


# ----------  Main experiment harness  --------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--trials", type=int, default=100,
                        help="number of independent data sets to test")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr",     type=float, default=1e-2)
    parser.add_argument("--atoms",  type=int, default=20)
    parser.add_argument("--inconsistent", action="store_true",
                        help="generate trace-INconsistent data instead")
    args = parser.parse_args([])

    perfect, losses = 0, []
    gen = synth_inconsistent if args.inconsistent else synth_consistent

    for t in range(args.trials):
        pos, neg = gen(num_atoms=args.atoms)
        acc, final_loss = run_trial(pos, neg, args.atoms,
                                    args.epochs, args.lr, verbose=False)
        losses.append(final_loss)
        if acc == 1.0:
            perfect += 1
        if (t + 1) % 10 == 0:
            print(f"Trial {t+1}/{args.trials}  "
                  f"accuracy {acc:.3f}  loss {final_loss:.4f}")

    print("\n================ Results ================")
    kind = "INCONSISTENT" if args.inconsistent else "CONSISTENT"
    print(f"Data type: {kind}")
    print(f"Trials with 100% training accuracy: {perfect}/{args.trials}")
    print(f"Mean final loss: {sum(losses)/len(losses):.4f}")
    print("=========================================")


if __name__ == "__main__":
    main()
