# CovSyn 重跑 firefly — 決策記錄

> **所有決策的總登記表在 `covsyn_decisions.md`**（本檔保留 D1–D12 的詳細背景與數據）。

> 基準狀態：6/30 04:41 的 run（200 代、best cost 0.8593）。
> 本次重跑於 2026-08-31 14:32 啟動（遠端 tmux session `firefly`，log `firefly_run4.log`）。

---

## 基準：上次跑的結果 vs reported mean 範圍

| 指標 | 6/30 跑出來 | reported mean 範圍 | 狀態 |
|---|---|---|---|
| Latent | 3.55 d | 4.10 – 5.50 | ✗ 低 0.55 d |
| Incubation | 5.18 d | 3.90 – 8.00 | ✓ |
| Infectious | 8.06 d | 3.45 – 20.00 | ✓ |
| Generation time | 5.85 d | 2.90 – 5.20 | ✗ 高 0.65 d |
| Serial interval | 5.82 d | 3.03 – 7.60 | ✓ |
| R0 | 0.68 | 0.30 – 6.70 | ✓ |

Cost 拆解（實測，與 `firefly_best.txt` 記錄值完全一致）：

| | 接觸人數 cost | 攻擊率 cost |
|---|---|---|
| Household | 0.0001 | 0.0114 |
| Health care | 0.2660 | 0.0030 |
| Others | 0.6716 | 0.0188 |
| **合計** | **0.9377 (96.6%)** | **0.0332 (3.4%)** |

各層 SAR（cheng2020 模式，10,000 index cases）：

| 層 | 合成 | 引用文獻 | 超出 |
|---|---|---|---|
| Household | 16.72% | 10.1% | 1.7× |
| School | 5.40% | 2.3% | 2.3× |
| Workplace | 25.13% | 3.4% | 7.4× |
| Health care | 1.67% | 0.4% | 4.2× |
| Municipality | 1.82% | 0.2% | 9.1× |

---

## 已決定並已實作

### D1 — latent vs generation time：方案 A ✅

模型中感染只能發生在 latent 之後（`Data_synthesize.py` 攻擊率向量前補 latent 個 0），故 `E[GT] ≥ E[latent]`；實測 offset = 2.30 d。兩個 reported mean 範圍（latent ≥ 4.10、GT ≤ 5.20）結構上無法同時滿足。

**決定：latent 進範圍、放掉 generation time。**
- latent 目標 4.1–4.5（reported mean 範圍下緣，讓 GT 超標幅度最小）
- 預期 GT ≈ 6.4 d，超出 5.20 上限約 1.2 d
- 論文需說明 Xin et al. 2022 的 latent 定義與 generation time 文獻不相容

### D2 — age risk ratio：鎖死 + 手動 try-and-error ✅

firefly 在正規化空間跑（`normalized*(ub-lb)+lb`），「沒有 bounds」在架構上不存在。改為 `lb == ub` 鎖死，每組試值改一次重跑。

**對齊目標改為 Cheng 的「所有感染」版本，不是 clinical 版本。** Cheng 的 2.19/1.75 排除了無症狀次級病例，而 CovSyn 計數所有感染、且無症狀比例與年齡無關（單一參數），結構上做不出 clinical 的年齡梯度。用 Cheng 自己的原始計數重算：

| | 計算 | SAR | RR |
|---|---|---|---|
| 0-19 | 1/281 | 0.356% | **0.52** |
| 20-39 | 8/1161 | 0.689% | 1 (ref) |
| 40-59 | 10/794 | 1.259% | **1.83** |
| 60+ | 3/331 | 0.906% | **1.32** |

**設定值 `[0.5, 1, 1.83, 1.32]`**。0-19 的 0.5 另有獨立佐證（Zhang 0.34、Davies 0.40、Viner 0.56、Uthman wild-type 0.58、Madewell 0.59，見 `covsyn_age_weight_litreview.md`）。舊值 0.3 的註解寫「by 1/281」，但 1/281 = 0.0036，推導不成立。

20-39 維持 1 作為錨點：加了 /Z 正規化後整組值只固定相對形狀，需要 pin 一組以消除尺度退化。

同時移除 `firefly_optimizer.py` 裡拉向 Cheng 的 RR penalty 項 —— 鎖死後它只是常數，但不同試值會產生不同常數，會讓各次 sweep 的 cost 不可比較。

### D3 — 加入 /Z 正規化（改為 per-layer）✅

`a_age = RR_age × ā` 改為 `a_age = (RR_age / Z_layer) × ā`，`Z_layer = Σ p_age^layer × RR_age`。
使各層的人口平均攻擊率維持在校準值（人口平均正規化為 1，OpenABM-Covid19 的慣例），同時保留年齡相對結構。

實測 Z：household / health care / municipality 1.2600、school 0.5588、workplace 1.4240。

### D4 — 記錄每個 contact 的年齡 ✅

新增 `{layer}_contact_ages`（每個接觸者一筆），保留原有 `{layer}_secondary_contact_ages`（僅感染者，未感染為 NaN）不變以免破壞既有分析。驗證：兩者長度一致。

### D5 — physiology penalty 目標改為 reported mean 範圍 ✅

- latent 3.0–4.0 → **4.1–4.5**
- incubation 5.0–6.0 → **5.2–8.0**
- infectious 7.0–10.0 → **5.0–10.0**
- 移除 age RR 四項（見 D2）
- 其餘（asymptomatic/symptomatic/critical 恢復時間、symptom→ICU 機率）維持

### D6 — generation time / serial interval / R0 **不**加進 cost function ❌

方案 A 已接受 GT 超出範圍，加進 cost 會與 D1 的決定衝突。維持事後驗證。

### D7 — 初始族群 seed 文獻中心值 ✅

firefly #0 設為文獻中心值（course of disease 用記錄值，contact 區塊用 bounds 中點），其餘 99 隻維持隨機。

### D8 — 各層攻擊率改為「累積 SAR → 每日機率」換算 ✅

**根因**：`household_attack_rate = 相對風險 × 0.101`，0.101 是文獻的**整體** household SAR，但被當成**每日**機率使用。每個接觸者平均有多天接觸，累積後大幅超標。

改為 `p_daily = 1 - (1 - SAR_cumulative)^(1/n_days)`，n_days 用前一次 run 實測值（household 2.00、school 2.95、workplace 4.02、health care 1.99、municipality 3.91）。

| 層 | 錨定累積 SAR | 新每日值 | 舊每日值 |
|---|---|---|---|
| household | 6.62%（Cheng 全感染 10/151；lb 4.6% Cheng clinical、ub 10.1% Huang 2021） | 0.0337 | 0.1030 |
| school | 2.3% | 0.0079 | 0.0277 |
| workplace | 3.4% | 0.0086 | 0.0695 |
| health care | 0.86%（Cheng 全感染 6/698） | 0.0043 | 0.0092 |
| municipality | 0.2% | 0.0005 | 0.0055 |

### D9 — 家庭 SAR 錨定 Cheng 的「所有感染」6.62% ✅

Cheng clinical 4.64% 對 Huang 10.1% 看似差 2.2 倍，但換成同一定義後 Cheng 全感染是 6.62%，只差 1.5 倍。取 Cheng 為中心值、Huang 為上界，兩個文獻都保留在搜尋範圍內。

### D10 — 不為 Cheng 的「Others」0.1% 放寬 bounds ❌

Cheng 的 Others 是 1836 個接觸、**僅 1 個病例**（CI 0–0.3%），且其分類（家戶／非家戶親屬／醫療／其他）中的「其他」比較接近 municipality，不含學校職場那種反覆密切接觸。各層錨自己的文獻，論文說明定義不對等。

用各層文獻值推算，S+W+M 合計應為 0.51%（Cheng 0.1%）。

### D11 — cost function 權重：攻擊率項乘 λ ✅

攻擊率被全域 max（16.7%）正規化，小攻擊率的層幾乎沒有份量。加 `ATTACK_RATE_WEIGHT`：

- 兩族項在舊最佳點等權需 λ ≈ 28
- **採 λ = 10** 作為保守的第一步（平衡會隨擬合改變而移動）
- 同時把 contact / attack_rate / energy / penalty 四項分開寫入 `progress_metrics.csv`，跑的過程中就能看到平衡

per-layer 正規化（把全域 max 換成各層 max）**不採用**：Others 的 max 只有 0.6%，會從一個極端跳到另一個極端。二項分佈 deviance 是更正規的做法，列為後續改進。

### D12 — 各層接觸者年齡分布 ✅

原本五層共用全人口分布。改為 school ∝ `age_p × student_p`、workplace ∝ `age_p × employment_p`；household / health care / municipality 維持全人口（輸入資料沒有這三層的年齡結構，家戶資料只有人數）。

| 層 | 0-19 | 20-39 | 40-59 | 60+ |
|---|---|---|---|---|
| 全人口（原本五層共用） | 0.167 | 0.266 | 0.317 | 0.250 |
| school | 0.882 | 0.118 | 0 | 0 |
| workplace | 0.008 | 0.441 | 0.493 | 0.058 |

實測驗證：school 接觸者 88.9% 為 0-19、workplace 48.8% 為 40-59。

⚠️ 這會改變合計量到的 RR（各層年齡組成與各層攻擊率不同，產生 Simpson 效應）。Cheng 的合計 RR 也有同樣的混淆，所以這樣反而更可比 —— 但代表 RR 的 try-and-error 必須在此之後進行。

---

## 實作前的驗證結果（seed 參數、關閉 overdispersion）

| 層 | 模擬 SAR | 錨定目標 | 每接觸天數 |
|---|---|---|---|
| Household | 8.94% | 6.62% | 5.30 |
| School | 2.37% | 2.30% | 5.49 |
| Workplace | 3.14% | 3.40% | 5.74 |
| Health care | 0.78% | 0.86% | 2.71 |
| Municipality | 0.12% | 0.20% | 3.62 |

⚠️ 換算用的每接觸天數取自**前一次最佳解**（household 2.00 等）；在 seed 參數下實際是 5.30，所以 household 偏高。這是起點的近似，firefly 會再調整接觸參數。

overdispersion 開在 bounds 中點（rO 0.10、wO 10.5，期望倍率 1.95）時所有層約略加倍 —— 這只是 seed 值，優化器會自己找。

---

## 尚未解決 / 待決定

- **循環論證** —— age RR 是輸入參數，量出來自然接近輸入值。論文需定位為 consistency check，或改用其他資料學 RR 再拿 Cheng 驗證。
- **0-19 的 clinical RR = 0 無法重現** —— 需要「年齡相關的有症狀比例」，是模型擴充（參數量 198 → 202）。
- **Cheng 的 RR 統計上不顯著**（2.19 的 CI 為 0.78–6.14，含 1）—— 只能在論文揭露。
- **家庭層年齡分布無資料** —— 輸入資料只有家戶人數，沒有年齡結構。
- **`--mode result` 與 `--mode cheng2020` 量到的 household SAR 差 10.2% vs 17.7%（約 2σ）** —— 未證實是噪音還是真實差異。訓練用 result、驗證用 cheng2020，若差異為真則兩者不是同一件事。
- **論文的 Table S2–S9、S11 全部需要重出。**

---

## 延後處理（等 CovSyn 完成、跑方法比較時才做）

- **Endpoint / 截斷定義**（todo §12）：保留 84 天模擬期，實作教授 email 與 reviewer 意見中討論過的所有 endpoint 納入規則，對每一種規則跑所有 R0 方法並比較。需要使用者提供教授的 email 與被拒稿的 reviewer comments（兩個 repo 都找不到）。
- **重讀 reviewer comments**（todo §13）：逐條對應到論文章節或驗證實驗。
- **(a) 台灣首波疫情種子數敏感度分析**（28 → 約 36–48）與 **年齡 RR 試值 `[0.39, 1, 1.90, 1.44]` 寫回 `parameters_for_initialization.py`**：依「先驗證、不改模型」原則延後到 todo Phase C。試值結果保存在遠端 `RR_trials/trial1/`。
