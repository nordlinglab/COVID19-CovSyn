"""Mean number of DAYS a candidate contact is actually met, per layer.

This is the quantity MEAN_CONTACT_DAYS in sar_anchors.py holds, and the conversion from the
literature CUMULATIVE secondary attack rate to a per-day probability depends on it:
p_daily = 1 - (1 - SAR)**(1/n_days). It is a model OUTPUT, so it has to be re-measured after
every run; letting it go stale is how the anchors came to exist in three places with three
different sets of numbers (finding E53).

Definition: the row sums of {layer}_contacts_matrix (contacts x days, True where the contact
was met) averaged over every candidate contact of every case in the files read.

Usage: python measure_days.py <synthetic_data_dir> [more dirs...]
"""
import glob
import os
import sys

import numpy as np

KEY = {'household': 'household_contacts_matrix', 'school': 'school_class_contacts_matrix',
       'workplace': 'workplace_contacts_matrix', 'health_care': 'health_care_contacts_matrix',
       'municipality': 'municipality_contacts_matrix'}
MAX_FILES = 400


def main():
    for directory in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(directory, 'contact_data_*.npy')))[:MAX_FILES]
        total = {layer: [0, 0, 0] for layer in KEY}      # days, rows, rows met at least once
        for path in files:
            for record in np.load(path, allow_pickle=True):
                for layer, key in KEY.items():
                    matrix = record.get(key)
                    if matrix is None or getattr(matrix, 'size', 0) == 0:
                        continue
                    days = np.asarray(matrix, dtype=bool).sum(axis=1)
                    total[layer][0] += int(days.sum())
                    total[layer][1] += len(days)
                    total[layer][2] += int((days > 0).sum())
        print('===', os.path.basename(directory.rstrip('/')), '(%d files)' % len(files))
        for layer in KEY:
            day_sum, rows, met = total[layer]
            if rows == 0:
                print('  %-13s no candidate contacts' % layer)
                continue
            print('  %-13s rows %7d  mean days/contact %5.2f  among those met %5.2f  met-share %.2f'
                  % (layer, rows, day_sum / rows, day_sum / max(met, 1), met / rows))


if __name__ == '__main__':
    main()
