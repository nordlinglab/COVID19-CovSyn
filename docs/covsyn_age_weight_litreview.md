# Replacing the Age Risk Ratio with a Principled Age Weight — Concept + Literature Review

> Goal: stop using Cheng's age **risk ratio** as a free multiplier on the daily SAR
> (`Data_synthesize.py:925`, see [covsyn_risk_ratio_issues.md](covsyn_risk_ratio_issues.md)) and replace it
> with a literature-grounded **age weight** that has a clear definition and external bounds.

---

## 1. The key conceptual insight: split one weight into two

Cheng's "age-related **secondary clinical attack rate** risk ratio" actually bundles **two distinct
age effects**:

```text
clinical attack rate(age)  =  susceptibility(age)      ×  clinical fraction(age)
                              "do I get infected?"          "if infected, do I show symptoms?"
```

- **Susceptibility** is lower in children, roughly flat in adults — it belongs on the **per-contact
  infection probability** (exactly where `r_a` currently sits).
- **Clinical fraction** rises monotonically with age — it belongs on the **symptomatic/asymptomatic
  transition**, which CovSyn *already models separately* (`infection_to_recovered_transition_p`,
  `Data_synthesize.py:350`), but currently age-independently.

**Why `r_a` "looks" monotonic increasing (0.3 → 2.19):** because it is a *clinical* attack-rate ratio =
susceptibility × clinical fraction. Children: low × low → very low. Elderly: ~flat × high → high.
Baking both into one SAR multiplier is the root confusion. The principled fix is to **split** them:

| Replaces `r_a` with | Where it applies in code | Literature basis |
|---|---|---|
| **σ(age)** age-specific *susceptibility* weight | per-contact infection prob (`calculate_age_adjusted_secondary_attack_rate`, L925) | Zhang 2020, Davies 2020, Viner 2021, Franco 2022 |
| **p_clin(age)** age-specific *clinical fraction* | symptom-onset transition (L350-352) | Davies 2020 |

If you want a **minimal** change (keep CovSyn's "clinical SAR" definition and one weight only), then keep a
single weight `w(age) = σ(age)·p_clin(age)` on the SAR — but **fixed to literature**, normalized, not freely
fitted. The full split is cleaner and is recommended.

---

## 2. Literature review — age-specific susceptibility to SARS-CoV-2 infection

All values are **relative susceptibility to infection given contact** (the quantity that should multiply the
per-contact probability). Reference group differs by study; noted per row.

| Study | Design | Age group | Relative susceptibility | 95% CI | Reference |
|---|---|---|---|---|---|
| **Zhang et al. 2020, *Science*** | Hunan contact tracing + RT-PCR (gold standard for susceptibility) | <15 | **0.34** | 0.24–0.49 | 15–64 = 1 |
| | | 15–64 | 1.00 | — | (ref) |
| | | >65 | **1.47** | 1.12–1.92 | 15–64 = 1 |
| **Davies et al. 2020, *Nat. Med.*** | age-structured model fit, 6 countries | 0–9 | **0.40** | 0.25–0.57 | adults |
| | | 60–69 | 0.88 | 0.70–0.99 | adults |
| | | <20 (summary) | ≈ 0.5 | — | ≈ half of 20+ |
| **Viner et al. 2021, *JAMA Pediatr.*** | meta-analysis, 32 studies | <20 vs ≥20 | **0.56** (odds) | 0.37–0.85 | ≥20 = 1 |
| **Franco et al. 2022, *PLOS Comput. Biol.*** (Belgium) | serial serology + contact data fit | [0,6) | **0.182** | 0.146–0.230 | [18,30) = 1 |
| | | [6,12) | 0.550 | 0.427–0.629 | [18,30) = 1 |
| | | [12,18) | 0.603 | 0.536–0.700 | [18,30) = 1 |
| | | [18,30) | 1.00 | 0.83–1.25 | (ref) |
| | | older adults | "decreases slightly" | — | [18,30) = 1 |

**Consensus:** children clearly less susceptible (≈0.2–0.6, strongest below 10y). Adults ≈1.
**Disagreement on elderly:** Zhang (contact-tracing) finds **higher** susceptibility in >65 (1.47);
Davies/Franco (model fits) find roughly **flat or slightly lower**. → the 60+ weight should carry **wide
bounds**.

## 2b. Newer papers (2023–2024) and a critical variant caveat

A fresh search for newer / Taiwan papers returned the following.

**Uthman et al. 2024, *PLoS One* — household-contact meta-analysis, susceptibility by variant** (the most
useful *new* source; resolves the elderly/variant uncertainty for the **right era**):

| Variant | Age group vs adults 20+ | Susceptibility OR | 95% CI |
|---|---|---|---|
| **Wild-type** (= CovSyn's Taiwan 2020 scenario) | 0–9 | 0.72 | 0.49–1.05 |
| | 10–19 | 0.77 | 0.68–0.88 |
| | **0–19 combined** | **0.58** | **0.44–0.77** |
| Alpha | 0–19 | 1.10 | 0.93–1.30 |
| Delta | 0–19 | 0.91 | 0.86–0.96 |

→ **Wild-type 0–19 ≈ 0.58 (0.44–0.77)** tightly corroborates Viner (0.56) and Zhang (0.34) — so the
0–19 susceptibility weight is well-anchored around **~0.5** for the 2020 period.

**⚠ Critical caveat — age susceptibility is variant-specific.** The child deficit **shrinks or reverses**
with newer variants: Alpha 0–19 OR ≈ 1.10, and Omicron-era studies (e.g. Toyama, Japan, *CDC EID* 2023;
Taiwan 2022–2023 cohorts below) show children becoming **highly** susceptible in households (Omicron
household OR up to ~6× vs pre-VOC). **CovSyn models Taiwan 2020 = wild-type, so use the wild-type values
(children ~0.5), NOT the Omicron-era ones.**

**Newer Taiwan papers found — but Omicron-era (wrong variant for CovSyn 2020):**
- *Epidemiological characteristics of the three waves of COVID-19 in Taiwan, Apr 2022–Mar 2023* (BA.2/BA.5/BA.2.75).
- *Taiwan household cohort (May 2022–Nov 2023)*: overall household SAR **37.2%** (Omicron), lower for
  vaccinated index (34.9% vs 63.2%) and prior-infection index (27.0% vs 46.3%).

  These are **Omicron**, dominated by vaccination/immunity effects, and are **not** appropriate for the
  2020 wild-type age weight. They are useful only if CovSyn is later extended to an Omicron scenario.

**Bottom line of the new search:** no new *Taiwan, wild-type, age-stratified susceptibility* paper exists
beyond Cheng 2020 (§4c). The best **new** input is the variant-stratified **Uthman 2024** meta-analysis,
which (for wild-type) tightens the 0–19 weight to ~0.5 and confirms §4a. Recommendation unchanged; the
0–19 bound can be tightened to **[0.34, 0.77]** using Zhang (lower) and Uthman wild-type (upper).

## 2c. Cross-country heterogeneity — does it differ a lot by country?

**Shape: consistent across countries. Level: varies a lot, but mostly NOT because of country.**

- **Madewell 2020 (household meta, 54 studies, 77,758 people):** adult contacts **28.3%** (20.2–37.1) vs
  child contacts **16.8%** (12.3–21.7) → adults ≈ **1.7×** children (child:adult ≈ 0.59). **No significant
  difference between China (21 studies) and other countries (33).**
- High heterogeneity (Madewell I² = 96.8% adults / 78.9% children; Viner I² = 94.6%) is driven by
  **variant, testing intensity, case definition (clinical vs PCR), contact setting, household size** — not
  by country per se.

Child:adult susceptibility ratio across very different settings clusters at **~0.34–0.59** (Zhang 0.34,
Davies 0.40, Viner 0.56, Uthman wild-type 0.58, Madewell 0.59) → the **age shape is portable**; only the
absolute level differs.

---

## 2d. Consolidated WILD-TYPE age profile (the relevant one for CovSyn = Taiwan 2020)

Two different "age distributions" — keep them separate:

### A. Relative **susceptibility** to infection (wild-type), reference 20–39 = 1

| Source (country) | 0–19 | 20–39 | 40–59 | 60+ |
|---|---|---|---|---|
| Zhang 2020 (China) | 0.34 (0.24–0.49)¹ | 1 (ref 15–64) | 1 | 1.47 (1.12–1.92)² |
| Davies 2020 (6 countries) | ≈0.40–0.50 (0–9: 0.40) | ~1 | ~0.9 | 0.88 (0.70–0.99)³ |
| Viner 2021 (32-study meta) | 0.56 (0.37–0.85) | 1 | 1 | — |
| Uthman 2024 (household meta) | 0.58 (0.44–0.77) | 1 | 1 | — |
| Franco 2022 (Belgium) | 0.18–0.60⁴ | 1 (ref 18–30) | ~0.9 | slightly <1 |
| Madewell 2020 (household meta) | child:adult ≈0.59 | 1 | 1 | — |
| **Synthesis (recommended)** | **≈0.50** (0.34–0.77) | **1.00** (anchor) | **≈0.95** (0.80–1.10) | **≈1.10** (0.70–1.90, uncertain) |

¹ <15 vs 15–64 · ² >65 vs 15–64 · ³ 60–69 · ⁴ strong within-band gradient: <6 ≈0.18, 12–18 ≈0.60

**Pattern:** children ~half of adults (strong gradient: toddlers ≈0.2, teens ≈0.6); adults flat ≈1;
elderly **uncertain** (China↑ 1.47 vs model fits ≈0.9).

### B. **Clinical (symptomatic) attack-rate** ratio (wild-type) — what Cheng/Taiwan measures

This = susceptibility × clinical fraction, so it rises more steeply with age (elderly symptomatic fraction
is higher). Reference 20–39 = 1.

| Source | 0–19 | 20–39 | 40–59 | 60+ |
|---|---|---|---|---|
| **Cheng 2020 (Taiwan), clinical SAR %** | 0% (0/281) | 0.5% | 1.1% | 0.9% |
| **Cheng ratio vs 20–39** | ~0 (floor 0.3) | 1 | **2.2** | **1.8** |
| Implied by susceptibility×clinical-fraction | low×low → very low | 1 | ↑ | ↑↑ |

→ This is exactly CovSyn's current `[0.3, 1, 2.19, 1.75]`. It is a **clinical** ratio (not susceptibility),
which is why it climbs with age and why splitting it (§1) is the clean fix.

### C. Age-specific **clinical fraction** (wild-type, Davies) — the second piece

| 0–19 | 20–39 | 40–59 | 60+ |
|---|---|---|---|
| ~0.21 (0.12–0.31) | ~0.35 | ~0.50 | ~0.69 (0.57–0.82) |

**Putting it together (wild-type):** clinical_SAR_ratio(age) ≈ susceptibility(age) × clinical_fraction(age) / (value at 20–39).
- 0–19: 0.50 × 0.21 → very low ✓ (Taiwan 0)
- 40–59: 0.95 × 0.50 → elevated ✓ (Taiwan 2.2)
- 60+: 1.10 × 0.69 → elevated ✓ (Taiwan 1.8)

The decomposition reproduces the Taiwan clinical pattern — confirming the §1 split is internally consistent.

---

## 2e. Susceptibility WITHOUT anchoring an age group to 1

**Methodological reality first:** age susceptibility is *intrinsically relative* — you cannot measure an
"absolute susceptibility given one contact" directly, because every estimate needs a scale. The scale is
either (a) an age group set to 1, or (b) the **population mean** set to 1, or you fall back to (c) an
**absolute observable** (per-contact SAR or seroprevalence) that conflates susceptibility with contacts /
exposure. So "no age = 1" really means **option (b) or (c)**.

### Option (b) — normalized to POPULATION MEAN = 1 (no single age pinned)
- **OpenABM-Covid19 (Hinch et al. 2021)** — `relative_susceptibility` explicitly *"normalized so that the
  average susceptibility for an individual in the population is 1."* Values [0.35, 0.69, 1.03×4, 1.27, 1.52,
  1.52], no single age = 1. → **This is exactly the `/Z` normalization in the CovSyn plan.**

> **Key point for the implementation:** the `ã_l^a = (σ[g_c]/Z)·ã_l` change (Z = Σ age_p·σ) makes the
> **population mean = 1, not any single age**. So the plan already satisfies your "no age = 1" preference —
> it is the OpenABM convention, not the "20–39 = 1" convention.

### Option (c) — ABSOLUTE infection-by-age (serology / per-contact), no reference at all
These report raw infection % by age (then you derive relative shape yourself):

- **Pollán et al. 2020, *Lancet* (ENE-COVID, Spain, n≈61,000)** — seroprevalence 0–19 yr **3.4–3.8%** vs
  **4.4–6.0%** in adults; <10 yr **<3.1%**. Absolute, nationwide.
- **Stringhini et al. 2020, *Lancet* (Geneva serosurvey)** — seropositive **0.8%** (5–9 yr) vs **9.6%**
  (10–19) vs **9.9%** (20–49). Absolute, by age.
- **Jing et al. 2020, *Lancet Infect Dis* (Guangzhou household)** — absolute household SAR **5.2%** (<20),
  **14.8%** (20–59), **18.4%** (≥60).

⚠ Serology/SAR absolute numbers fold in **exposure and contact patterns**, so they are not pure
susceptibility — use them as **shape evidence / validation**, not as the per-contact weight directly.

### Newer susceptibility-parameter papers found (for completeness)
- **Boldea et al. 2024, *PNAS Nexus*** — age-specific susceptibility by **variant** (Netherlands); children
  much lower than adults until Omicron, then rising. (Anchors adults >19 = 1; variant-stratified, like
  Uthman 2024.)
- **Hu et al. 2021, *Nature Communications* (Hunan, 1178 infectors / 15,648 contacts)** — contact-adjusted
  susceptibility **increases with age**; infectiousness not age-different.
- **Goldstein, Lipsitch & Cevik 2021, *J Infect Dis*** — q-susceptibility review: children ~**20–50%** of
  adult susceptibility.

**Takeaway:** there is no paper giving "absolute susceptibility" with no scale at all (impossible by
definition). The clean "no single age = 1" choice is **population-mean normalization (OpenABM)** — which is
already what the CovSyn `/Z` plan does. Absolute serology/SAR papers (Pollán, Stringhini, Jing) are the
best "no reference age" *observational* evidence, used for shape/validation.

## 3. Literature — age-specific clinical (symptomatic) fraction

Use this for the **symptom-onset transition**, not the SAR multiplier.

| Study | Age group | Clinical fraction | 95% CI |
|---|---|---|---|
| **Davies et al. 2020, *Nat. Med.*** | 10–19 | **0.21** | 0.12–0.31 |
| | (monotonic rise with age) | | |
| | 70+ | **0.69** | 0.57–0.82 |

Pattern: symptomatic fraction **increases monotonically** with age, ~0.2 (children) → ~0.7 (elderly).

---

## 4. Proposed CovSyn weights and Firefly bounds (4 groups: 0–19, 20–39, 40–59, 60+)

Reference group = **20–39 (locked at 1)**. Bounds are taken from the union of the CIs above, slightly
widened for the elderly given the cross-study disagreement.

### 4a. Susceptibility weight σ(age) — replaces `r_a` on the infection probability

| Age group | Point (init) | Lower bound | Upper bound | Rationale |
|---|---|---|---|---|
| 0–19 | **0.45** | 0.34 | 0.77 | Zhang 0.34, Viner 0.56, **Uthman 2024 wild-type 0.58 (0.44–0.77)** |
| 20–39 | **1.00** | 1.00 | 1.00 | reference, **locked** |
| 40–59 | **0.95** | 0.70 | 1.15 | adults ≈1, slight decrease (Davies/Franco) |
| 60+ | **1.10** | 0.70 | 1.92 | Zhang 1.47(–1.92) vs Davies 0.88 → **wide** |

> Apply **mean-preserving** so it does not shift the calibrated base SAR:
> `σ_norm(age) = σ(age) / Z`, with `Z = Σ_age p_age·σ(age)` (population-weighted mean). This is the missing
> `/Z` from [covsyn_risk_ratio_issues.md](covsyn_risk_ratio_issues.md), now applied to a *susceptibility*
> weight whose meaning ("relative to 20–39") is well defined.

### 4b. Clinical fraction p_clin(age) — for the symptom-onset transition (optional, fuller fix)

| Age group | Point | Lower | Upper | Source |
|---|---|---|---|---|
| 0–19 | 0.21 | 0.12 | 0.35 | Davies 10–19 = 0.21 (0.12–0.31) |
| 20–39 | 0.35 | 0.25 | 0.50 | Davies interpolation |
| 40–59 | 0.50 | 0.40 | 0.62 | Davies interpolation |
| 60+ | 0.69 | 0.57 | 0.82 | Davies 70+ = 0.69 |

This would make `infection_to_recovered_transition_p` (= P(asymptomatic)) age-dependent =
`1 − p_clin(age)`, instead of the current single value.

---

## 4c. Taiwan-specific data (the source CovSyn already uses)

There **is** Taiwan age-stratified data — it is exactly Cheng et al. 2020, the study CovSyn calibrates on.
But it measures the **clinical attack rate** (susceptibility × clinical fraction), not pure susceptibility,
and the sample is tiny.

**Cheng et al. 2020, *JAMA Intern. Med.* (Taiwan, Table 2) — secondary clinical attack rate by contact age:**

| Age group | Contacts | Secondary cases | Clinical attack rate (95% CI) | Ratio vs 20–39 |
|---|---|---|---|---|
| 0–19 | 281 | 1 (asymptomatic) | **0%** (0–1.4%) | 0 → CovSyn floors at 0.3 |
| 20–39 | 1,161 | 8 | **0.5%** (0.2–1.1%) | 1 (reference) |
| 40–59 | 794 | 10 | **1.1%** (0.6–2.1%) | **2.2** → CovSyn 2.19 |
| 60+ | 331 | 3 | **0.9%** (0.3–2.6%) | **1.8** → CovSyn 1.75 |
| Overall | 2,761 | 22 | 0.7% (0.4–1.0%) | — |
| Household | 151 | — | 4.6% (2.3–9.3%) | — |

→ This is **exactly** where CovSyn's `[0.3, 1, 2.19, 1.75]` initial values and the Table S3 bounds came from
(Cheng's 95% CIs converted to ratios; 0–19 set to 0.3 = 1/281 because it had **0 clinical** cases).

**Huang, Tu & Lai 2021, *J. Microbiol. Immunol. Infect.* (Taiwan nationwide meta-analysis):** overall SAR
0.84% (95% CrI 0.42–1.69%) — **not age-stratified**, useful only for the overall level.

**Why Taiwan-only data cannot pin the 4 age weights:**
- Only **22 secondary cases total**; 0–19 had **0 clinical** cases (the 0.3 is a floor, not an estimate);
  60+ rests on **3 cases**. CIs are enormous → the optimizer is essentially unconstrained → it drifts to
  `[0.026, 0.138, 0.927, 0.469]`.
- It is a **clinical** ratio, so its rise with age partly reflects the **clinical fraction**, not
  susceptibility — consistent with the split in §1.

**Recommended use of Taiwan vs international data:**
- Use **Cheng (Taiwan)** as the **level / validation target** for the *clinical* attack rate by age
  (it is the local ground truth, and matches the CovSyn cost data).
- Borrow the **age *shape*** (susceptibility σ and clinical fraction p_clin) from the **larger
  international studies** (Zhang/Davies/Viner/Franco, thousands of events) as an **informative prior /
  bounds**, because Taiwan alone is too sparse to identify four numbers.
- Concretely: fix/normalize σ(age) from §4a (international shape), make p_clin(age) from §4b
  (Davies), then **validate** the resulting simulated clinical attack rate by age against Cheng's
  `0 / 0.5 / 1.1 / 0.9 %`.

## 5. How this fixes the original problems

| Original problem (risk_ratio_issues) | Fixed by |
|---|---|
| relative ratio used as absolute multiplier, no `/Z` | σ is explicitly relative-to-20–39 **and** normalized by `Z` (§4a) |
| free param drifts directionally wrong | σ **fixed/bounded to literature CIs**; 20–39 locked as anchor |
| reference group not locked | 20–39 = 1 enforced |
| clinical-vs-susceptibility conflation | split into σ (infection) and p_clin (symptoms) |
| no age-stratified validation | σ and p_clin become **validation targets** computable from outputs (simulated susceptibility ratio, simulated clinical fraction by age) |

---

## 6. Implementation pointers

- **Susceptibility weight:** rename/repurpose `age_risk_ratios` → `age_susceptibility` and change
  `calculate_age_adjusted_secondary_attack_rate` (`Data_synthesize.py:924-927`) to
  `ã_l^a = min(1, (σ[g_c]/Z) · ã_l)`; precompute `Z = Σ age_p·σ` in `__init__`.
- **Bounds:** set in `parameters_for_initialization.py:935-940` from §4a (lock 20–39 lb=ub=1).
- **Clinical fraction (optional):** make `transition_p[0]` age-dependent using `secondary_contact_age`/case
  age in `draw_course_of_disease` (`Data_synthesize.py:350`).
- **Re-run Firefly** (bounds changed). If σ is fully fixed, this removes 4 free dimensions; if kept
  learnable, add an age-stratified SAR cost term (else it drifts again).

---

## 7. Caveats / open questions

- **Susceptibility vs infectiousness:** the weight here is infectee-side **susceptibility** — correct for
  `r_a`'s position. Infector-side infectiousness is more uncertain (Franco: "very large CIs") and is already
  handled by the day-since-infection SAR curve; do not add a second age-infectiousness term without data.
- **Elderly susceptibility is genuinely uncertain** (Zhang↑ vs Davies/Franco flat). Keep 60+ bounds wide,
  or treat the 60+ value as the one learnable parameter with an age cost.
- **Definition of CovSyn's base SAR:** if it stays a *clinical* (symptomatic) attack rate and you do **not**
  split out p_clin, then use the combined `w(age)=σ·p_clin` weight instead of pure σ — but fix it, don't fit
  it freely.
- **Per-layer age distribution:** σ is sampled at the contact's age `g_c ~ age_p` (global). School/workplace
  age composition differs; `Z` ideally computed per layer (see [covsyn_risk_ratio_issues.md](covsyn_risk_ratio_issues.md) Problem 6).

---

## Sources

- [Zhang et al. 2020, *Science* — Changes in contact patterns shape the dynamics of the COVID-19 outbreak in China](https://www.science.org/doi/10.1126/science.abb8001) (susceptibility <15: 0.34 [0.24–0.49]; >65: 1.47 [1.12–1.92])
- [Davies et al. 2020, *Nature Medicine* — Age-dependent effects in the transmission and control of COVID-19 epidemics](https://www.nature.com/articles/s41591-020-0962-9) (susceptibility 0–9: 0.40 [0.25–0.57]; clinical fraction 10–19: 0.21 → 70+: 0.69)
- [Viner et al. 2021, *JAMA Pediatrics* — Susceptibility to SARS-CoV-2 Infection Among Children and Adolescents Compared With Adults](https://pmc.ncbi.nlm.nih.gov/articles/PMC7519436/) (<20 vs ≥20 odds 0.56 [0.37–0.85])
- [Franco et al. 2022, *PLOS Comput. Biol.* — Inferring age-specific differences in susceptibility to and infectiousness upon SARS-CoV-2 infection (Belgian data)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009965) (ref [18,30)=1; [0,6): 0.182; [6,12): 0.550; [12,18): 0.603)

### Susceptibility without anchoring an age to 1 (population-mean / absolute)
- [Hinch et al. 2021, *PLOS Comput Biol* — OpenABM-Covid19 (relative_susceptibility normalized to population mean 1)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009146); [parameter doc](https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/infection_parameters.md)
- [Pollán et al. 2020, *Lancet* — ENE-COVID nationwide seroprevalence study, Spain](https://www.sciencedirect.com/science/article/pii/S0140673620314835) (absolute seroprevalence by age: 0–19 yr 3.4–3.8%)
- [Stringhini et al. 2020, *Lancet* — Geneva serosurvey](https://pmc.ncbi.nlm.nih.gov/articles/PMC7546669/) (5–9 yr 0.8% vs 20–49 yr 9.9%)
- [Boldea et al. 2024, *PNAS Nexus* — Age-specific transmission dynamics over first 2 years (variant-stratified susceptibility)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10837015/)
- [Hu et al. 2021, *Nature Communications* — Infectivity, susceptibility, and risk factors under intensive contact tracing in Hunan](https://www.nature.com/articles/s41467-021-21710-6) (susceptibility increases with age)
- [Goldstein, Lipsitch & Cevik 2021, *J Infect Dis* — On the effect of age on transmission (q-susceptibility)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7386533/)

### Newer (2023–2024)
- [Uthman et al. 2024, *PLoS One* — Susceptibility and infectiousness of SARS-CoV-2 in children versus adults, by variant (wild-type, Alpha, Delta): household-contact meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11379298/) (wild-type 0–19 susceptibility OR 0.58 [0.44–0.77]; age effect is variant-specific)
- [Variants & Age-Dependent Infection Rates among Household and Nonhousehold Contacts, *CDC EID* 2023 (Toyama, Japan)](https://wwwnc.cdc.gov/eid/article/29/8/22-1582_article) (Omicron: children's household infection rose to ~38%; age pattern flips by variant)
- [Epidemiological characteristics of three COVID-19 waves in Taiwan, Apr 2022–Mar 2023 (Omicron)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10213295/)
- [Taiwan household cohort, vaccinated index cases & transmission (May 2022–Nov 2023, Omicron)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975059/) (household SAR 37.2%)

### Taiwan-specific (2020, wild-type — matches CovSyn)
- [Cheng et al. 2020, *JAMA Internal Medicine* — Contact Tracing Assessment of COVID-19 Transmission Dynamics in Taiwan](https://pmc.ncbi.nlm.nih.gov/articles/PMC7195694/) (Table 2 age-stratified clinical SAR: 0–19: 0%; 20–39: 0.5%; 40–59: 1.1%; 60+: 0.9%; overall 0.7%; household 4.6%) — **the source of CovSyn's `[0.3,1,2.19,1.75]`**
- [Huang, Tu & Lai 2021, *J. Microbiol. Immunol. Infect.* — Estimation of the secondary attack rate of COVID-19 using nationwide contact-tracing data in Taiwan](https://pmc.ncbi.nlm.nih.gov/articles/PMC7289119/) (overall SAR 0.84%, 95% CrI 0.42–1.69%; not age-stratified)
