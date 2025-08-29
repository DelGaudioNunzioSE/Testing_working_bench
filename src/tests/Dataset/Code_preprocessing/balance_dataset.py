from datasets import Dataset
from collections import defaultdict
import random

def balanced_sample_multi_cols(
    ds: Dataset,
    cols=("language", "source", "variant"),
    desired_n: int = 1000,
    seed: int = 30,
):
    """
    Returns a balanced sub-dataset across ALL combinations of `cols`,
    with the same number of examples for each combination.  
    If 1000 cannot be reached while keeping perfect balance, it returns 
    the maximum balanced size <= desired_n.

    Note: only the combinations that ACTUALLY exist in the dataset are considered.
    """
    rng = random.Random(seed)

    # 1) index dataset examples by group (tuple of column values)
    idx_by_group = defaultdict(list)
    for i, ex in enumerate(ds):
        key = tuple(ex[c] for c in cols)
        idx_by_group[key].append(i)

    groups = list(idx_by_group.keys())
    G = len(groups)
    if G == 0:
        raise ValueError("Nessun gruppo trovato: controlla i nomi delle colonne.")

    # 2) how many can we take per group while keeping equality?
    #    (a) theoretical target is floor(desired_n / G)
    #    (b) cannot exceed the smallest group size
    target_per_group = desired_n // G
    cap_min = min(len(idxs) for idxs in idx_by_group.values())
    take = min(target_per_group, cap_min)

    if take == 0:
        raise ValueError(
            f"Cannot create a balanced sample: desired_n={desired_n} < number of groups={G} "
            f"or some groups have 0 elements. Increase desired_n or reduce group dimensions."
        )

    # 3) campiona esattamente 'take' per ogni gruppo
    chosen = []
    for g in groups:
        idxs = idx_by_group[g]
        chosen.extend(rng.sample(idxs, k=take))

    chosen.sort()
    sub = ds.select(chosen)

    return sub