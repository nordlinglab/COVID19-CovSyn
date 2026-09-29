"""Extract per-layer external references from the Taiwan COVID-19 contact-tracing workbooks
(Wu & Nordling structured course-of-disease dataset) for CovSyn validation (todo.md 2.4, 2.5, 5).

Relationship columns are mapped to CovSyn layers with the table recorded as F9 in
covsyn_decisions.md. A relationship that belongs to several layers for the same pair is
assigned by priority household > school > workplace > health_care > municipality.

Outputs (validation_reference/):
  taiwan_case_timeline.csv               one row per case with the durations the workbook dates
                                         both ends of (infection to onset, onset to
                                         confirmation) plus the mid-point of the reported age
                                         band, so the per-case distributions can be compared
                                         without shipping the workbook itself
  taiwan_tracing_contacts_per_case.csv   one row per (dataset, index case, layer) with any record:
                                         infected = distinct case IDs listed in the layer's columns,
                                         uninfected = sum of the layer's 'number of uninfected contact'
                                         columns (empty if none recorded)
  taiwan_infection_events.csv            one row per case with a recorded infector
                                         ('source infected case'), with the relationship to the
                                         infector and the mapped layer

Usage: python -m covsyn.data_processing.extract_tracing_reference [forecasting_repo] [output_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('../Repositories/nordlinglab-covid19-forecasting')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('validation_reference')
TAIWAN = REPO / 'Data' / 'Taiwan'
SOURCES = [
    ('first_wave_2020', TAIWAN / 'A_structured_course_of_disease_dataset_with_contact_tracing_information_in_Taiwan_for_COVID-19_modelling' / 'taiwan_covid_figshare.xlsx'),
    ('extended_to_2021', TAIWAN / 'covidtable_taiwan_translated.xlsx'),
]

# relationship column -> (CovSyn layer or None, matching 'number of uninfected contact' suffix or None)
RELATIONSHIPS = {
    'couple': ('household', 'couple'),
    'parent/child': ('household', None),
    'grandparent/grandchild': ('household', None),
    'brother/sister': ('household', None),
    'family': ('household', 'family'),
    'live_together': ('household', 'live_together'),
    'the_same_school': ('school', 'school'),
    'coworker': ('workplace', 'coworker'),
    'the_same_hospital': ('health_care', 'hospital'),
    'friend': ('municipality', 'friend'),
    'other_(unknown)_contact': ('municipality', 'others'),
    'the_same_flight': (None, 'flight'),
    'the_same_flight_(nearby_seat)': (None, 'flight_nearby_seats'),
    'travel_together': (None, 'travel'),
    'the_same_car': (None, 'car'),
    'the_same_hotel': (None, 'hotel'),
    'the_same_quarantine_hotel': (None, 'quarantine_hotel'),
    'panshi_fast_combat_support_ship': (None, 'panshi'),
    'coral_princess': (None, None),
}
LAYER_PRIORITY = ['household', 'school', 'workplace', 'health_care', 'municipality']


def case_ids(cell):
    """Case IDs in a cell such as 'ID 42, 45' or 'ID 745~748 755'; 'C'/'X'/empty -> none."""
    text = str(cell).strip()
    if text.lower() in ('c', 'x', 'nan', ''):
        return set()
    ids = set()
    for a, b in re.findall(r'(\d+)\s*~\s*(\d+)', text):
        ids |= {int(i) for i in range(int(a), int(b) + 1)}
    return ids | {int(i) for i in re.findall(r'\d+', re.sub(r'\d+\s*~\s*\d+', '', text))}


def layer_of(relationships):
    layers = {RELATIONSHIPS[r][0] for r in relationships}
    return next((L for L in LAYER_PRIORITY if L in layers), None)


# Contacts recorded only in the free text of the CDC press release, not in the structured columns (F8).
# Case 601: "已掌握個案接觸者共3人，為公司同事，因曾與個案共餐，列居家隔離" (confirmed 2020-11-13, imported).
TEXT_ONLY = [{'dataset': 'extended_to_2021', 'case_id': 601, 'case_type': 'Abroad', 'confirmed_date': '2020-11-13',
              'layer': 'workplace', 'infected': 0, 'uninfected': 3, 'total': 3, 'from_text': True}]

contacts, events = [], []
for dataset, path in SOURCES:
    df = pd.read_excel(path, sheet_name='Individual_data')
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
    df['case_type'] = df['abroad/local'].astype(str).str.strip().str.capitalize()
    by_id = df.set_index('id')

    for i in df.index:
        for layer in LAYER_PRIORITY:
            columns = [r for r, (L, _) in RELATIONSHIPS.items() if L == layer]
            infected = set().union(*(case_ids(df.at[i, r]) for r in columns))
            counts = [pd.to_numeric(df.at[i, f'number_of_uninfected_contact_({RELATIONSHIPS[r][1]})'], errors='coerce')
                      for r in columns if RELATIONSHIPS[r][1]]
            counts = [c for c in counts if pd.notna(c)]
            if not infected and not counts:
                continue
            uninfected = int(sum(counts)) if counts else None
            contacts.append({'dataset': dataset, 'case_id': int(df.at[i, 'id']), 'case_type': df.at[i, 'case_type'],
                             'confirmed_date': pd.to_datetime(df.at[i, 'confirmed_date'], errors='coerce').date(),
                             'layer': layer, 'infected': len(infected), 'uninfected': uninfected,
                             'total': len(infected) + (uninfected or 0), 'from_text': False})

        for source in sorted(case_ids(df.at[i, 'source_infected_case'])):
            me = int(df.at[i, 'id'])
            relationships = {r for r in RELATIONSHIPS if source in case_ids(df.at[i, r])}
            if source in by_id.index:
                relationships |= {r for r in RELATIONSHIPS if me in case_ids(by_id.at[source, r])}
            events.append({'dataset': dataset, 'infectee_id': me, 'infectee_type': df.at[i, 'case_type'],
                           'source_id': source,
                           'source_type': by_id.at[source, 'case_type'] if source in by_id.index else '',
                           'relationships': ';'.join(sorted(relationships)),
                           'layer': layer_of(relationships) or 'no_covsyn_layer'})

OUT.mkdir(parents=True, exist_ok=True)

# Per-case timeline: the two durations whose endpoints are both dated in the workbook, and the
# age band. Used by plot_violin_reality_vs_covsyn.py, which needs the observed DISTRIBUTIONS.
AGE_BAND = {'0 to 9': 5, '10 to 19': 15, '20 to 29': 25, '30 to 39': 35, '40 to 49': 45,
            '50 to 59': 55, '60 to 69': 65, '70 to 79': 75, '80 to 89': 85, '90 to 99': 95}
timeline = []
for dataset, path in SOURCES:
    df = pd.read_excel(path, sheet_name='Individual_data')
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')
    if 'onset_of_symptom' not in df.columns:
        continue
    onset = pd.to_datetime(df['onset_of_symptom'], errors='coerce')
    confirmed = pd.to_datetime(df['confirmed_date'], errors='coerce')
    # The second workbook does not carry an earliest-infection date at all.
    infected = (pd.to_datetime(df['earliest_infection_date'], errors='coerce')
                if 'earliest_infection_date' in df.columns else pd.Series(pd.NaT, index=df.index))
    for i in df.index:
        timeline.append({
            'dataset': dataset,
            'case_id': int(df.at[i, 'id']),
            'case_type': str(df.at[i, 'abroad/local']).strip().capitalize(),
            'age_band_midpoint': AGE_BAND.get(str(df.at[i, 'age']).strip()),
            'infection_to_onset': (onset[i] - infected[i]).days if pd.notna(onset[i]) and pd.notna(infected[i]) else None,
            'onset_to_confirmation': (confirmed[i] - onset[i]).days if pd.notna(onset[i]) and pd.notna(confirmed[i]) else None,
            'symptomatic': bool(pd.notna(onset[i])),
        })
pd.DataFrame(timeline).to_csv(OUT / 'taiwan_case_timeline.csv', index=False)

contacts = pd.DataFrame(contacts + [r for r in TEXT_ONLY
                                    if not any(c['dataset'] == r['dataset'] and c['case_id'] == r['case_id']
                                               and c['layer'] == r['layer'] for c in contacts)])
events = pd.DataFrame(events)
contacts.to_csv(OUT / 'taiwan_tracing_contacts_per_case.csv', index=False)
events.to_csv(OUT / 'taiwan_infection_events.csv', index=False)

for dataset, g in contacts.groupby('dataset', sort=False):
    print(f'--- contacts per case, {dataset}')
    print(g.groupby('layer')['total'].agg(['count', 'mean', 'median', 'max']).round(2).to_string())
for dataset, g in events.groupby('dataset', sort=False):
    local = g[g['infectee_type'] == 'Local']
    print(f'--- infection events, {dataset}: {len(g)} with a recorded infector ({len(local)} local infectees)')
    print(pd.concat({'all': g['layer'].value_counts(), 'local': local['layer'].value_counts()}, axis=1).fillna(0).astype(int).to_string())
print('saved', OUT / 'taiwan_tracing_contacts_per_case.csv', ',', OUT / 'taiwan_infection_events.csv',
      'and', OUT / 'taiwan_case_timeline.csv', f'({len(timeline)} cases)')
