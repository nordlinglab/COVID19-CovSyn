"""Rebuild school_p so it is weighted by PUPILS, not by schools (finding E39).

get_school_data() builds, for each grade, a histogram of "students per class" with one entry
per SCHOOL. A school holds several classes, so the class size a random STUDENT experiences is
distributed proportionally to pupils = class size x number of classes, not to class size
alone. Decision B30 applied a class-size weighting at load time, which closed only two thirds
of the gap (18.2 -> 22.6 against the 25.7 a first-grader actually sees) because large schools
have both bigger classes and more of them (correlation 0.64).

This script rewrites school_p inside variable/demographic_parameters.pkl using pupil weights,
reading the same Ministry of Education files and the same columns as get_school_data(). It is
idempotent: the result depends only on the raw files, not on the previous contents.

Usage: python -m covsyn.calibration.rebuild_school_pmf [data_dir] [variable_dir]
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/demographic_data')
VAR = Path(sys.argv[2] if len(sys.argv) > 2 else 'variable')
DROP = ['金門縣', '連江縣', '澎湖縣']
# (file, first class column, first student column, grades, first age, rows to drop)
SOURCES = [('國民小學校別資料.xls', 6, 15, 6, 7, None),
           ('國民中學校別資料.xlsx', 6, 12, 3, 13, None),
           # Senior high: the file also carries the attached junior-high divisions, which
           # get_school_data() drops before building the histogram, so drop them here too.
           ('高級中等學校校別資料檔.xls', 9, 16, 3, 16, ('學程(等級)別', 'J'))]

# University (ages 19-22) is a different shape and needs its own pass: the file is one row
# per department and grade, so the row IS the cohort and there is no class column. That is
# why this script originally skipped it -- and that was wrong (finding E49). A per-row
# histogram gives every DEPARTMENT equal weight, but a random STUDENT lands in a department
# with probability proportional to its size, so the same pupil weighting applies. It used to
# be supplied at load time by apply_person_weighting(); decision E39 removed the school layer
# from that function wholesale, which silently dropped university weighting too and took the
# mean university class size from 87.51 down to 59.92 against Taiwan's 88.33.
UNIVERSITY = ('大專校院各校科系別學生數.xlsx', 9, 4, 19)   # file, first column, grades, first age


def read(name):
    df = pd.read_excel(DATA / name)
    df.columns = df.iloc[1]
    df = df[2:]
    return df[~df['縣市名稱'].isin(DROP)] if '縣市名稱' in df.columns else df


def pupil_weighted_pmf(sizes, pupils, length):
    """Histogram of class size weighted by the number of pupils in those classes."""
    pmf = np.zeros(length)
    index = np.clip(np.round(sizes).astype(int), 0, length - 1)
    np.add.at(pmf, index, pupils)
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def main():
    with open(VAR / 'demographic_parameters.pkl', 'rb') as f:
        demo = pickle.load(f)
    school_p = demo[7]
    backup = VAR.parent / 'ARCHIVE_20260923_run_phaseD2' / 'demographic_parameters_pre_E39.pkl'
    if backup.parent.exists() and not backup.exists():
        shutil.copy(VAR / 'demographic_parameters.pkl', backup)

    before, after = {}, {}
    for name, class_pos, student_pos, grades, first_age, drop_rows in SOURCES:
        try:
            df = read(name)
        except FileNotFoundError:
            print('skipped (file missing):', name)
            continue
        if drop_rows is not None:
            column, value = drop_rows
            df = df[df[column] != value]
        for i in range(grades):
            age = first_age + i
            if age not in school_p:
                continue
            classes = pd.to_numeric(df.iloc[:, class_pos + i], errors='coerce')
            pupils = (pd.to_numeric(df.iloc[:, student_pos + 2 * i], errors='coerce')
                      + pd.to_numeric(df.iloc[:, student_pos + 2 * i + 1], errors='coerce'))
            keep = (classes > 0) & (pupils > 0)
            sizes = (pupils[keep] / classes[keep]).to_numpy(dtype=float)
            weights = pupils[keep].to_numpy(dtype=float)
            old = np.asarray(school_p[age], dtype=float)
            grid = np.arange(len(old))
            before[age] = float((grid * old).sum() / max(old.sum(), 1e-12))
            new = pupil_weighted_pmf(sizes, weights, len(old))
            school_p[age] = new
            after[age] = float((grid * new).sum())

    name, first_col, grades, first_age = UNIVERSITY
    try:
        df = read(name)
        df = df[df['等級別'] == 'B 學士']
        df = df[df['縣市名稱'] != '71 金門縣']
    except FileNotFoundError:
        print('skipped (file missing):', name)
        df = None
    if df is not None:
        for i in range(grades):
            age = first_age + i
            if age not in school_p:
                continue
            cohort = (pd.to_numeric(df.iloc[:, first_col + 2 * i], errors='coerce')
                      + pd.to_numeric(df.iloc[:, first_col + 2 * i + 1], errors='coerce'))
            cohort = cohort[cohort > 0].to_numpy(dtype=float)
            old = np.asarray(school_p[age], dtype=float)
            grid = np.arange(len(old))
            before[age] = float((grid * old).sum() / max(old.sum(), 1e-12))
            # sizes and weights are the same array: the cohort size IS the number of
            # students who experience that cohort size.
            new = pupil_weighted_pmf(cohort, cohort, len(old))
            school_p[age] = new
            after[age] = float((grid * new).sum())

    demo[7] = school_p
    with open(VAR / 'demographic_parameters.pkl', 'wb') as f:
        pickle.dump(demo, f)

    print('mean class size per grade (age: per-school histogram -> pupil weighted)')
    for age in sorted(after):
        tag = '  <- university (E49)' if age >= first_age else ''
        print('  age %2d   %6.2f -> %6.2f%s' % (age, before[age], after[age], tag))


if __name__ == '__main__':
    main()
