# CovSyn 機率鏈流程圖 (Flowcharts)

> 以下兩張 Mermaid 流程圖直接對應 `Data_synthesize.py` 的實際程式碼路徑，
> 並標出可疑節點（risk ratio 當成乘數的問題）。

---

## Flowchart A — CovSyn 完整參數→輸出工作流程

```mermaid
flowchart TD
    A["輸入參數<br/>attack_rate / age_risk_ratios / overdispersion<br/>contact P 向量 / vaccination_rate"] --> B["人口與 agent 生成<br/>DemographicData / SocialData"]
    B --> C["社交層結構<br/>household / school / workplace<br/>health_care / municipality<br/>(social_size 各層)"]

    A --> D["疾病病程抽樣<br/>draw_course_of_disease() (L314)<br/>latent / incubation / infectious<br/>monitor_isolation_period"]

    C --> E["draw_contact_data(P) (L931)"]
    D --> E

    E --> F["draw_contacts_each_day(P) (L747)<br/>每層接觸矩陣<br/>generate_logistic_contact_p (L471)"]
    E --> G["calculate_daily_secondary_attack_rate() (L816)<br/>把每層 attack_rate 曲線<br/>對齊到 latent/incubation/infectious 天數<br/>→ adjusted_attack_rate (逐日)"]

    F --> H["逐一接觸者迴圈 (L946+)<br/>draw_from_previously_infected_set<br/>draw_vaccination_status<br/>secondary_contact_age ~ age_p"]
    G --> H

    H --> I["draw_infection_status() (L884)<br/>(見 Flowchart B)"]
    I --> J{"感染成功?"}
    J -- "是" --> K["記錄 effective_contact<br/>感染時間 / 年齡<br/>population_size -= 1"]
    J -- "否" --> H

    K --> L["疾病進展<br/>symptom / critically ill / recovery / death"]
    L --> M["合成輸出資料<br/>confirmed / recovered / dead<br/>secondary contact ages"]
    M --> N["估計 R0 / ground-truth R0<br/>世代鏈"]
```

---

## Flowchart B — Infector → Infectee 機率鏈

```mermaid
flowchart TD
    I0["Infector i (已感染)<br/>病程: latent/incubation/infectious"] --> I1["距感染天數 t"]
    I1 --> I2["逐日傳染力曲線<br/>adjusted_attack_rate[t]<br/>calculate_daily_secondary_attack_rate (L816)"]

    L0["社交層 l<br/>household/school/workplace<br/>health_care/municipality"] --> L1["每日接觸機率<br/>generate_logistic_contact_p (L471)<br/>healthy_p / symptom_p / steepness"]
    L1 --> L2["contact_day_vector (是否接觸)<br/>draw_social_contacts_each_day (L680)"]

    J0["Infectee j (候選接觸者)"] --> J1["secondary_contact_age ~ age_p (L967)"]
    J1 --> J2["age_risk_ratios[age]"]
    J0 --> J3["natural_immunity_status / vaccination_status"]

    %% draw_infection_status (L884)
    J3 --> S0{"免疫或已接種?<br/>(L887)"}
    S0 -- "是" --> SX["P = 0 → 不感染"]
    S0 -- "否" --> S1

    I2 --> S1
    J2 --> S1
    S1["age_adjusted_attack_rate<br/>= age_risk_ratios[age] × adjusted_attack_rate<br/>calculate_age_adjusted_secondary_attack_rate (L924-927)"]:::warn

    S1 --> S2["× contact_day_vector (L895)<br/>= effective_attack_rate"]
    L2 --> S2

    S2 --> S3{"overdispersion 狀態?<br/>determine_overdispersion_state (L810)"}
    S3 -- "是" --> S4["p = min(1, rate × overdispersion_weight) (L902)"]
    S3 -- "否" --> S5["p = effective_attack_rate"]

    S4 --> S6["逐日 random < p (L909-912)<br/>取第一個感染日 (L915)"]
    S5 --> S6
    S6 --> S7["感染事件 i → j"]

    classDef warn fill:#ffe0e0,stroke:#c00,stroke-width:2px;
```

---

## 圖中標紅節點（S1）= 核心疑慮

`calculate_age_adjusted_secondary_attack_rate` (L924-927) 做的是：

```python
age_adjusted_attack_rate = self.age_risk_ratios[secondary_contact_age] * adjusted_attack_rate
```

- ✅ **正確的部分**：risk ratio 是套在 **infectee（secondary_contact_age）** 的年齡上，符合「次級臨床侵襲率風險比應對應被感染者年齡」的預期。
- ⚠️ **疑慮部分**：`age_risk_ratios` 被當成一個**獨立乘數 (input weight)** 直接乘進機率鏈。
  若文獻中的 age risk ratio 本身是「整條機率鏈的最終 emergent 結果」，那麼在這裡再乘一次就會造成
  **重複計入 / 機率定義衝突** —— 應該是驗證目標 (validation target)，而非鏈內乘數。
