# What CovSyn's `r_a` Should Become — and What Peer ABMs Put There

> Reframed question: not "what is the age risk ratio", but **"in CovSyn's equation, what is that slot, and
> what do comparable COVID agent-based models put in the same slot?"**

---

## 1. Where `r_a` sits in CovSyn (the slot)

[Data_synthesize.py:925](Data_synthesize.py#L925):
```python
age_adjusted_attack_rate = self.age_risk_ratios[secondary_contact_age] * adjusted_attack_rate
```
`r_a` is a **multiplicative factor on the per-contact infection probability, indexed by the age of the
person being infected (the susceptible/infectee).**

In standard ABM terminology this slot has a name: **age-specific relative susceptibility** (often
`rel_sus` / `relative_susceptibility` / `S_a`). It is *not* a "risk ratio of the secondary attack rate";
it is the susceptible's relative probability of being infected given an exposure.

So: **replace `age_risk_ratios` with `relative_susceptibility(age)`** — same position in the equation, but a
quantity with a clear definition and an established modelling convention.

---

## 2. What peer ABMs use in that exact slot (大家的值)

Both models CovSyn compares itself to (Supplementary Table S10) have this parameter, **both sourced from
Zhang et al. 2020**:

### Covasim (Institute for Disease Modeling) — `rel_sus`, 10-yr brackets
| Age | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80–89 | 90+ |
|---|---|---|---|---|---|---|---|---|---|---|
| rel_sus | 0.34 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.24 | 1.47 | 1.47 | 1.47 |

Applied as a multiplicative factor on infection risk per exposure event. Source: Zhang 2020. Reference
group = 20–59 (= 1).

### OpenABM-Covid19 (Oxford / BDI-pathogens) — `relative_susceptibility`, 10-yr brackets
| Age | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80+ |
|---|---|---|---|---|---|---|---|---|---|
| rel_sus | 0.35 | 0.69 | 1.03 | 1.03 | 1.03 | 1.03 | 1.27 | 1.52 | 1.52 |

Source: Zhang 2020. **Explicitly normalized so the population-mean susceptibility = 1** — i.e. exactly the
`/Z` normalization that CovSyn is missing.

**Both are:** (a) susceptibility, not clinical attack rate; (b) monotonic ↑ with age (children low, elderly
high); (c) from Zhang 2020; (d) the same equation slot as CovSyn's `r_a`.

---

## 3. Map peer convention onto CovSyn's 4 age groups

Collapsing the 10-yr brackets to CovSyn's `[0–19, 20–39, 40–59, 60+]`:

| CovSyn group | from Covasim | from OpenABM | **Recommended rel_sus** |
|---|---|---|---|
| 0–19 | (0.34, 0.67) → ~0.50 | (0.35, 0.69) → ~0.52 | **0.50** |
| 20–39 | 1.00 | 1.03 | **1.00** (reference) |
| 40–59 | 1.00 | 1.03 | **1.00** |
| 60+ | (1.24–1.47) → ~1.40 | (1.27–1.52) → ~1.44 | **1.40** |

Then **normalize to population mean 1** (OpenABM convention): `rel_sus_norm(age) = rel_sus(age) / Z`,
`Z = Σ age_p · rel_sus(age)`.

---

## 4. The key contrast with CovSyn's current values

| | 0–19 | 20–39 | 40–59 | 60+ | Quantity | Shape |
|---|---|---|---|---|---|---|
| **CovSyn `r_a` (current)** | 0.3 | 1 | **2.19** | **1.75** | clinical SAR ratio | peaks at 40–59 then dips |
| **Covasim / OpenABM rel_sus** | ~0.5 | 1 | ~1.0 | **~1.4** | susceptibility | monotonic ↑ with age |

CovSyn's `[0.3, 1, 2.19, 1.75]` is a **clinical attack-rate ratio** (susceptibility × symptom fraction),
which is why it bulges at 40–59 and is non-monotonic. Peer ABMs **do not** put that in the susceptibility
slot — they put pure susceptibility (monotonic, Zhang) there and model the **symptomatic/clinical age
effect separately** (Covasim `prognoses` → age-specific symptomatic probability; OpenABM age-specific
fraction asymptomatic). That is the same split recommended in
[covsyn_age_weight_litreview.md](covsyn_age_weight_litreview.md) §1.

---

## 5. Recommendation (peer-aligned, minimal)

1. Rename `age_risk_ratios` → `relative_susceptibility`; values `[0.50, 1.0, 1.0, 1.40]` (Zhang 2020 via
   Covasim/OpenABM), reference 20–39 = 1.
2. Apply with population-mean normalization (`/Z`) like OpenABM:
   `ã_l^a = min(1, (rel_sus[g_c]/Z) · ã_l)`.
3. Move the age **clinical/symptomatic** effect to the symptom-onset transition
   (`infection_to_recovered_transition_p`), age-dependent — Davies clinical fraction
   ([covsyn_age_weight_litreview.md](covsyn_age_weight_litreview.md) §3).
4. Keep `[0.50, 1.0, 1.0, 1.40]` **fixed** (or only let 60+ vary, with wide bounds), since the elderly value
   is the one genuinely debated (Zhang/Covasim/OpenABM ↑1.4–1.5 vs Davies/Franco ≈0.9).
5. Validate the resulting **clinical** attack rate by age against Cheng Taiwan `0 / 0.5 / 1.1 / 0.9 %`.

---

## Sources
- [Covasim — InstituteforDiseaseModeling/covasim `parameters.py` (rel_sus default)](https://github.com/InstituteforDiseaseModeling/covasim/blob/main/covasim/parameters.py); [Kerr et al. 2021, PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149) — rel_sus [0.34, 0.67, 1,1,1,1, 1.24, 1.47, 1.47, 1.47], from Zhang 2020
- [OpenABM-Covid19 — infection_parameters.md (relative_susceptibility, normalized to mean 1)](https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/infection_parameters.md); [Hinch et al. 2021, PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009146) — rel_sus [0.35, 0.69, 1.03×4, 1.27, 1.52, 1.52], from Zhang 2020
- [Zhang et al. 2020, Science — original age-susceptibility estimates (<15: 0.34; >65: 1.47)](https://www.science.org/doi/10.1126/science.abb8001)
