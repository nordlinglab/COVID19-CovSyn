"""Is the city ratio check (B27, 'largest / smallest city mean community contacts' in [1, 1.6])
measuring a real city effect, or the noise of a max / min over many small groups?

The community contact count is drawn as Poisson(P[28] * nu) with no city term (B27), so a model
with no city effect at all would still show some spread between city means. This takes the index
cases of a spread dataset exactly as verify_phaseD.py does, computes the observed ratio, and then
the same ratio after shuffling the city labels 2,000 times -- the distribution of the ratio when
city and contacts are independent by construction. If the observed ratio sits inside that null
distribution, the check fails on sampling noise, not on a city effect.

Usage: python probe_city_ratio.py [spread_dir]
"""
import glob
import sys
from pathlib import Path

import numpy as np

SPREAD = Path(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data_results_spread_Taiwan_weight')
MIN_CASES = 20          # verify_phaseD.py only uses cities with at least 20 index cases


def ratio(community, city):
    means = [community[city == c].mean() for c in np.unique(city) if (city == c).sum() >= MIN_CASES]
    return max(means) / max(min(means), 1e-9), len(means)


def main():
    community, city = [], []
    for f in sorted(glob.glob(str(SPREAD / 'contact_data_*.npy'))):
        k = f.split('_')[-1].split('.')[0]
        contact = np.load(f, allow_pickle=True)
        social = np.load(SPREAD / f'social_data_{k}.npy', allow_pickle=True)
        if len(contact) and len(social):
            community.append(len(contact[0]['municipality_effective_contacts'] or []))
            city.append(social[0]['municipality'])
    community, city = np.asarray(community, float), np.asarray(city)
    observed, n_cities = ratio(community, city)
    sizes = sorted((int((city == c).sum()) for c in np.unique(city) if (city == c).sum() >= MIN_CASES))
    print(f'{len(community)} index cases, {n_cities} cities with >= {MIN_CASES} cases, sizes {sizes}')
    print(f'overall mean {community.mean():.2f}, sd {community.std():.2f}')
    print(f'observed largest / smallest city mean: {observed:.3f}')

    rng = np.random.default_rng(0)
    null = np.array([ratio(community, rng.permutation(city))[0] for _ in range(2000)])
    print(f'shuffled city labels (no city effect by construction), 2,000 times:')
    print(f'  median {np.median(null):.3f}, 5-95% [{np.percentile(null, 5):.3f}, {np.percentile(null, 95):.3f}]')
    print(f'  share of shuffles <= 1.6 (would pass the check): {np.mean(null <= 1.6):.3f}')
    print(f'  share of shuffles >= observed {observed:.3f}: {np.mean(null >= observed):.3f}')


if __name__ == '__main__':
    main()
