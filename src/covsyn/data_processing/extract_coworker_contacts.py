"""Extract coworker close contacts per index case from the Taiwan COVID-19 contact-tracing
workbooks (Wu & Nordling structured course-of-disease dataset), as an exploratory external
reference for CovSyn workplace candidate contacts.

For every index case with any coworker record:
  infected_coworkers   number of case IDs listed in the 'coworker' column
  uninfected_coworkers 'number_of_uninfected_contact_(coworker)' (missing -> not recorded)
  total                infected + uninfected (a missing uninfected count is taken as 0)

Usage: python -m covsyn.data_processing.extract_coworker_contacts [forecasting_repo] [output_csv]
"""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('../Repositories/nordlinglab-covid19-forecasting')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('validation_reference/taiwan_coworker_close_contacts.csv')
TAIWAN = REPO / 'Data' / 'Taiwan'
SOURCES = [
    ('first_wave_2020', TAIWAN / 'A_structured_course_of_disease_dataset_with_contact_tracing_information_in_Taiwan_for_COVID-19_modelling' / 'taiwan_covid_figshare.xlsx'),
    ('extended_to_2021', TAIWAN / 'covidtable_taiwan_translated.xlsx'),
]


def count_ids(cell):
    text = str(cell).strip()
    if text.lower() in ('c', 'x', 'nan', ''):
        return 0
    return len(re.findall(r'\d+', text))


rows = []
for dataset, path in SOURCES:
    df = pd.read_excel(path, sheet_name='Individual_data')
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
    infected = df['coworker'].map(count_ids)
    uninfected = pd.to_numeric(df['number_of_uninfected_contact_(coworker)'], errors='coerce')
    recorded = (infected > 0) | uninfected.notna()
    for i in df.index[recorded]:
        rows.append({'dataset': dataset,
                     'case_id': df.at[i, 'id'],
                     'case_type': str(df.at[i, 'abroad/local']).strip().capitalize(),
                     'confirmed_date': pd.to_datetime(df.at[i, 'confirmed_date'], errors='coerce').date(),
                     'infected_coworkers': int(infected[i]),
                     'uninfected_coworkers': None if pd.isna(uninfected[i]) else int(uninfected[i]),
                     'total_coworker_close_contacts': int(infected[i] + (0 if pd.isna(uninfected[i]) else uninfected[i]))})

out = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
for dataset, g in out.groupby('dataset', sort=False):
    t = g['total_coworker_close_contacts']
    print(f'{dataset}: {len(g)} index cases, mean {t.mean():.2f}, median {t.median():.1f}, '
          f'local {int((g["case_type"] == "Local").sum())}, imported {int((g["case_type"] != "Local").sum())}')
print('saved', OUT)
