"""Expand every literature value CovSyn compares itself against into one traceable table.

The values live as `report_*` dictionaries inside plot_result/plot_synthetic_data.ipynb, one
entry per study, and everything downstream (plot_10panel.py, the physiology penalty targets,
the ranges printed on the validation figures) is derived from them by taking minima and
maxima. That derivation hides two things the thesis has to state:

  * which study each number comes from, and
  * whether the number is a MEAN or a MEDIAN. The latent-period range 4.1-5.5, used as a
    calibration target throughout, is Cheng 2020's median (4.1) next to Xin 2022's mean
    (5.5); the infectious-period entries mix means, medians and fixed model parameters. A
    range built from mixed statistics is not a range of means.

Output: validation_reference/literature_provenance.csv, one row per (quantity, study), plus
a summary of how many studies and which statistic types stand behind each range.

Usage: python -m covsyn.data_processing.extract_literature_provenance [notebook] [out_csv]
"""
import ast
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK = Path(sys.argv[1] if len(sys.argv) > 1 else 'plot_result/plot_synthetic_data.ipynb')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'validation_reference/literature_provenance.csv')

# quantity -> (variable name in the notebook, what the three numbers mean)
QUANTITIES = {
    'latent period': 'report_latent_period',
    'incubation period': 'report_incubation_period',
    'infectious period': 'report_infectious_period',
    'asymptomatic infectious period': 'report_asymptomatic_infectious_period',
    'pre-symptomatic infectious period': 'report_presymptomatic_infectious_period',
    'post-symptomatic infectious period': 'report_postsymptomatic_infectious_period',
    'generation time': 'report_generation_time',
    'serial interval': 'report_serial_interval',
}
# Statistic type is only recorded in free-text comments, so it is matched by keyword.
STATISTIC = [('median', 'median'), ('mean', 'mean'), ('fixed', 'fixed parameter'),
             ('max', 'maximum'), ('iqr', 'median (IQR)'), ('range', 'median (range)')]


def notebook_source(path):
    cells = json.load(io.open(path, encoding='utf-8'))['cells']
    return ''.join(''.join(c.get('source', [])) for c in cells)


def literal_after(source, name):
    """The dict literal assigned to `name`, with per-entry trailing comments kept."""
    start = source.find(name + ' =')
    if start < 0:
        return None, None
    open_brace = source.find('{', start)
    depth, i = 0, open_brace
    while i < len(source):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    text = source[open_brace:i + 1]
    # the comment block immediately above the assignment usually carries the review citation
    head = source.rfind('\n#', max(0, start - 1200), start)
    context = source[head:start].strip() if head > 0 else ''
    return text, context


def parse(text):
    """Values keyed by study.

    Some entries are written as expressions rather than literals (np.sqrt(11.4) for a
    standard deviation, 15.7 - 6.7 for a difference), so literal_eval is not enough; the
    dictionary is evaluated with numpy available and nothing else."""
    clean = re.sub(r'#[^\n]*', '', text)
    try:
        return eval(clean, {'__builtins__': {}}, {'np': np, 'nan': np.nan})
    except Exception as exc:
        print('  could not parse:', exc)
        return {}


def statistic_of(study, entry_comment, quantity_context):
    """What kind of estimate this number is.

    Only an entry's OWN comment can settle it. A note in the block above the dictionary
    ("Cheng's estimates are median") applies to the study it names and to no other, so
    attributing it to every entry would invent information -- which is exactly the mixing
    of statistics the thesis has to avoid."""
    own = entry_comment.lower()
    for keyword, label in STATISTIC:
        if keyword in own:
            return label
    for sentence in re.split(r'[.\n]', quantity_context.lower()):
        if study.lower() in sentence:
            for keyword, label in STATISTIC:
                if keyword in sentence:
                    return label + ' (block note)'
    return 'not stated'


def main():
    source = notebook_source(NOTEBOOK)
    rows = []
    for quantity, variable in QUANTITIES.items():
        text, context = literal_after(source, variable)
        if text is None:
            print('missing:', variable)
            continue
        comments = dict(re.findall(r"'([^']+)'\s*:\s*\[[^\]]*\]\s*,?\s*#\s*([^\n]*)", text))
        for study, values in parse(text).items():
            values = list(values) + [None] * (3 - len(values))
            rows.append({
                'quantity': quantity,
                'notebook_variable': variable,
                'study': study,
                'central': values[0], 'low': values[1], 'high': values[2],
                'statistic_type': statistic_of(study, comments.get(study, ''), context),
                'entry_comment': comments.get(study, ''),
                'review_or_source_note': ' '.join(context.split())[-300:],
            })
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print('saved', OUT, f'({len(table)} study-level values)')

    print('\n%-36s %7s %9s %9s   %s' % ('quantity', 'studies', 'min mean', 'max mean', 'statistic types'))
    for quantity, group in table.groupby('quantity', sort=False):
        central = pd.to_numeric(group['central'], errors='coerce').dropna()
        kinds = sorted(set(group['statistic_type']))
        flag = '  <-- MIXED STATISTICS' if len([k for k in kinds if k != 'not stated']) > 1 else ''
        print('%-36s %7d %9.2f %9.2f   %s%s'
              % (quantity, len(group), central.min(), central.max(), ', '.join(kinds), flag))

    print('\nstudies reporting more than one of these quantities (the latent / generation-time question):')
    by_study = table.groupby('study')['quantity'].apply(lambda x: sorted(set(x)))
    shared = by_study[by_study.apply(len) > 1]
    if len(shared) == 0:
        print('  none')
    for study, quantities in shared.items():
        print('  %-14s %s' % (study, ', '.join(quantities)))
    latent = set(table[table['quantity'] == 'latent period']['study'])
    generation = set(table[table['quantity'] == 'generation time']['study'])
    print('\nlatent period studies      :', ', '.join(sorted(latent)))
    print('generation time studies    :', ', '.join(sorted(generation)))
    print('reporting BOTH             :', ', '.join(sorted(latent & generation)) or 'NONE')


if __name__ == '__main__':
    main()
