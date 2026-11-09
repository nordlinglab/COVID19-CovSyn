# CovSyn Probability-Chain Flowcharts (English, with paper notation)

> Symbols follow the Supplementary **Table S1** notation list, and the node logic follows
> Algorithms **S2** (main loop), **S9** (adjusted daily SAR), **S11** (draw infection status),
> **S12** (age-adjusted SAR), **S13/S14** (contacts each day), and Eq. (1) (contact-probability function).
> Code anchors point to `Data_synthesize.py`.

## Notation quick key

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| `q` | infection queue | `t_I` | time of infection |
| `t_L` | simulation time limit | `t_M` | time of monitored isolation |
| `τ_L / τ_I / τ_F` | latent / incubation / infectious period | `τ_G` | generation time |
| `n_H,n_S,n_W,n_C,n_M` | layer sizes (household/school/workplace/clinic/municipality) | `M` | contact matrix (all layers) |
| `a_l` | daily secondary attack rate (SAR) vector, layer `l` | `ã_l` | adjusted daily SAR vector |
| `ã_l^a` | age-adjusted daily SAR vector | `m` | contact day vector |
| `m_e` | effective contact day vector | `c` | number of contacts |
| `r_a[g_c]` | age risk ratio at secondary-contact age `g_c` | `g_c` | age of secondary contact |
| `s_N / s_V / s_O` | natural-immunity / vaccination / overdispersion state | `r_O / w_O` | overdispersion rate / weight |
| `f_pL` | contact-probability function (Eq. 1) | `p_h,p_s,s,t_PS,t_PN` | its parameters |

---

## Flowchart A — Full CovSyn parameter→output workflow (Algorithm S2)

```mermaid
flowchart TD
    A["Input parameters<br/>a_l, r_a, r_O, w_O, r_N, r_V, e_V<br/>contact-prob params p,p_c,p_h,p_s,s,t_PS,t_PN"] --> B["Init infection queue q, previously-infected set 𝕀, population set ℙ"]
    B --> C{"q not empty?"}
    C -- "no" --> Z["End → demographic, social, course-of-disease, contact data + transmission digraph"]
    C -- "yes" --> D["Pop case (source_id, t_I, prev_idx, g_c, layer)"]
    D --> E{"t_I ≤ t_L ?"}
    E -- "no" --> Z
    E -- "yes" --> F["Draw demographic data (Alg S5)<br/>age / gender / occupation"]
    F --> G["Draw social context (Alg S6)<br/>sizes n_H, n_S, n_W, n_C, n_M"]
    G --> H["Draw course of disease (Alg S3)<br/>τ_L, τ_I, τ_F, t_M, t_TP, t_R/t_IC/t_D"]
    H --> I["Draw social contact data (Alg S4)<br/>(see Flowchart B)"]
    I --> J["For each layer l, each effective contact:<br/>append (case_id, t_I+τ_G, prev_idx, g_c, l) to q"]
    J --> C
```

Code: main loop in [Data_synthesis_main.py](Data_synthesis_main.py); contact stage `draw_contact_data` [Data_synthesize.py:931](Data_synthesize.py#L931).

---

## Flowchart B — Infector → Infectee probability chain (Algorithms S4/S9/S11/S12)

```mermaid
flowchart TD
    subgraph SRC["Infector i (course of disease)"]
        I0["τ_L, τ_I, τ_F, t_M"] --> I1["Layer SAR vector a_l"]
        I1 --> I2["Adjusted daily SAR ã_l (Alg S9)<br/>stretch a_l to τ_F centered at τ_I,<br/>pad τ_L zeros front, trim/pad to t_M"]
    end

    subgraph LAY["Social layer l contact process (Alg S14)"]
        L0["Daily contact prob (Eq. 1)<br/>f_pL(t) = 2p_h − p_s − (p_h−p_s)/(1+e^(−s(t−t_PS))) − (p_h−p_s)/(1+e^(s(t−t_PN)))"] --> L1["c ~ Binomial(n_l, p)"]
        L1 --> L2["Contact matrix M_l → contact day vector m"]
    end

    subgraph TGT["Infectee j"]
        J0["g_c ~ age probabilities"] --> J1["risk ratio r_a[g_c]"]
        J0 --> J2["s_N (natural immunity), s_V (vaccination)"]
    end

    J2 --> S0{"s_N or s_V == True?<br/>(Alg S11 line 5)"}
    S0 -- "yes" --> SX["P = 0 → no infection"]
    S0 -- "no" --> S1

    I2 --> S1
    J1 --> S1
    S1["Age-adjusted daily SAR (Alg S12, L924-927)<br/>ã_l^a = r_a[g_c] × ã_l ; clip ã_l^a[ã_l^a>1]=1"]:::warn

    S1 --> S2["Effective rate per day: ã_l^a × m (L895)"]
    L2 --> S2

    S2 --> S3{"Overdispersion s_O ? (Alg S8, r_O)"}
    S3 -- "yes" --> S4["p_inf = min(1, ã_l^a[j] × w_O) (L902)"]
    S3 -- "no" --> S5["p_inf = ã_l^a[j]"]

    S4 --> S6["Bernoulli per day; first success → first infection day (L909-915)"]
    S5 --> S6
    S6 --> S7["s_I = True, m_e[j]=1 → infection event i→j<br/>append j to q at t_I + τ_G"]

    classDef warn fill:#ffe0e0,stroke:#c00,stroke-width:2px;
```

Code anchors: `calculate_daily_secondary_attack_rate` [Data_synthesize.py:816](Data_synthesize.py#L816);
`draw_infection_status` [Data_synthesize.py:884](Data_synthesize.py#L884);
`calculate_age_adjusted_secondary_attack_rate` [Data_synthesize.py:924](Data_synthesize.py#L924).

---

## The highlighted node S1 = the core concern

Algorithm **S12** / code [Data_synthesize.py:924-927](Data_synthesize.py#L924-L927):

```text
ã_l^a = ã_l × r_a[g_c]          (then clip values > 1 to 1)
```

- ✅ **Correct:** `r_a` is indexed by the **infectee's** age `g_c` (the secondary contact), matching the
  intended meaning "age-related secondary clinical attack-rate risk ratio refers to the infectee's age."
  In the code `g_c` is drawn from the age distribution at [Data_synthesize.py:967](Data_synthesize.py#L967) — no infector/infectee swap bug.
- ⚠️ **Concern:** `r_a` enters as an **independent multiplier inside** the probability chain. The
  trained values (Table S3: `r_a(0-19)=0.03`, `r_a(20-39)=0.14`, `r_a(40-59)=0.93`, `r_a(60+)=0.47`)
  scale the per-day SAR directly. If the literature age risk ratio is meant to be the **emergent cumulative
  result** of the whole chain (Eq. 7, ρ = x/n) rather than an input weight, multiplying here risks
  double-counting / a definitional conflict. It should arguably be a **validation target**, not a chain multiplier.

This is exactly the question Sections 3–4 of the audit prompt ask about.
```
