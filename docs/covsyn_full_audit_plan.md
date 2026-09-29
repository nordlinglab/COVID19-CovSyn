# CovSyn 全面審查與修正計畫（Full Audit Plan）

> 本文件整理對 CovSyn 的全面審查結果，涵蓋 code bug、參數設計缺陷、模型邏輯異常，以及尚待確認的面向。
> 審查目標：確保 Firefly 的 198 個參數值合理，且整個 CovSyn 模擬邏輯合理。

---

## 0. 重要前提：證據強度分級

在進行任何修改前，必須先理解每個問題的「證據強度」。並非所有列出的問題都已 100% 確認，部分需要核對外部資料（論文 Supplementary）或重跑模擬才能定案。

| 證據等級 | 定義 | 行動 |
|---|---|---|
| **A — 已直接驗證** | 我直接讀 code / 算參數值得到的事實 | 可直接依此判斷 |
| **B — 需核對 Supplementary** | 依賴診斷報告對論文 Supplementary 的引用，我未親自讀過 PDF | 改 code 前必須核對原文 |
| **C — 推理待驗證** | 基於邏輯推斷，需重跑模擬或進一步分析才能確認 | 列為待驗證項，勿直接改 |

**關於「Supplementary Algorithm S11」：** 這是指 CovSyn 論文補充附錄（Supplementary Appendix）的第 11 個演算法。**這份 PDF 不在 repository 內，本審查未親自讀過原文**，所有引用 S11 的結論均為證據等級 B，修改前須取得論文 Supplementary 核對。

---

## 1. 確認的 Code Bug

### Bug 1：Overdispersion 公式 `**` vs `*`

- **位置：** [Data_synthesize.py:902-903](Data_synthesize.py#L902-L903)
- **證據等級：** A（code 改動已確認）+ B（Supplementary 規格未親自核對）

**已確認的事實（等級 A）：**

| 來源 | 公式 | 效果（rate=0.05, weight=6.31） |
|---|---|---|
| 原始 commented-out code（line 869） | `rate * weight` | `0.05 × 6.31 = 0.315`（放大） |
| 目前 code（line 903） | `rate ** weight` | `0.05^6.31 ≈ 1.3e-8`（近乎歸零） |

code 確實在「Claude3.5 optimize」refactor 時從 `*` 改成 `**`。對所有 attack_rate < 1 的值，兩者效果**方向相反**（放大 vs 壓制），非數值微差。

**待核對的事實（等級 B）：**

診斷報告稱 Supplementary Algorithm S11（page 23）規定 `â_i[j] × w_O`（乘法 = 放大）。**修改 code 前必須親自取得論文 Supplementary 第 23 頁核對此公式。**

**修正方式（核對 Supplementary 為乘法後執行）：**

```python
# Data_synthesize.py line 902-903
overdispersion_p = np.minimum(1, effective_attack_rate * self.overdispersion_weight)
```

**修正後連帶影響：** `overdispersion_weight` 語意反轉（weight>1 變成放大），原 bounds [1, 20] 需重設（weight=20 在乘法下過大），建議 ub 降到 3。

---

### Bug 2：`draw_vaccination_status()` 使用 `overdispersion_rate`

- **位置：** [Data_synthesize.py:804-808](Data_synthesize.py#L804-L808)
- **證據等級：** A（code 事實確認）；**是否需修正：取決於設計意圖（B）**

**已確認事實：** `draw_vaccination_status()` 與 `determine_overdispersion_state()` 程式碼**完全相同**，兩者都用 `self.overdispersion_rate` 抽樣：

```python
def draw_vaccination_status(self):          # 決定二次接觸者是否免疫
    overdispersion_state = random.choices(
        [True, False], weights=[self.overdispersion_rate, 1-self.overdispersion_rate])[0]
    return (overdispersion_state)

def determine_overdispersion_state(self):    # 決定是否為 superspreader（內容一字不差）
    overdispersion_state = random.choices(
        [True, False], weights=[self.overdispersion_rate, 1-self.overdispersion_rate])[0]
    return (overdispersion_state)
```

`vaccination_rate`（P[68]）的 bounds 鎖死 [0, 0]，因此 `draw_vaccination_status()` 若改用正確的 `vaccination_rate`，效果是「vaccine_status 永遠 False」。

**這不是 vectorization 引入的 bug——原始 commented-out code 就已如此**（line 863 的 `vaccine_status` 來自 `draw_vaccination_status()`）。它是歷史遺留的命名/語意混亂。

**關鍵判斷：修不修正，行為差異如下**

| 做法 | 效果 |
|---|---|
| 維持現狀（用 overdispersion_rate） | 16.6% 二次接觸者被當成免疫而跳過感染（抑制傳播） |
| 改用正確 vaccination_rate（=0） | vaccine_status 永遠 False，等於移除這層抑制（R0 proxy 會再上升） |

**結論：** 這不是「明確必修的 bug」，而是「設計意圖未明的語意問題」。**先確認 Supplementary 中 vaccination 與 overdispersion 的關係**：
- 若 S11 明定 vaccine_status 應由 vaccination_rate 抽 → 是 bug，修正並同步處理 R0 上升。
- 若論文本就用單一 overdispersion_rate 同時建模 → 維持現狀，僅需在文件中註明此設計。

> 註：原審查報告將此標為「P0 必修」過於武斷。實際優先序取決於上述核對結果。

---

### Bug 3：Overdispersion state 從 per-day 改為 per-person

- **位置：** [Data_synthesize.py:898-906](Data_synthesize.py#L898-L906)
- **證據等級：** A（行為改變確認）+ B（S11 的 per-day 意圖未核對）

**已確認事實：** 原始 code 在 `for j in range(...)` 迴圈內，**每個接觸日獨立**抽 `overdispersion_state`（line 866）。Vectorize 後改為整個人**只抽一次**（line 898），全部天數共用同一狀態。

| | 原始 code | 目前 code |
|---|---|---|
| 抽樣頻率 | 每個接觸日各抽一次 | 每人只抽一次 |
| 生物學意義 | 每次暴露事件獨立具 superspreader 可能 | 此人整體是/否 superspreader |

**與 Bug 1/Bug 2 的關係：是三個獨立的問題，不是同一個 bug。**
- Bug 2 = 用錯 rate（哪個變數）。
- Bug 3 = 抽樣的粒度（per-day vs per-person）。
- Bug 1 = 公式運算子（`*` vs `**`）。

**優先序：** 在 Bug 1 未修正前，overdispersion 整體在「壓制」感染，per-day 與 per-person 都是壓制，量級差異小，影響被掩蓋。**建議先修 Bug 1，重跑診斷後再評估 Bug 3 是否需還原為 per-day。** 還原與否須先核對 S11 的 `j` 下標確為日期索引（等級 B）。

---

### Bug 4：`school_previously_infected_index_list` 拼字錯誤

- **位置：** [Data_synthesis_main.py:69](Data_synthesis_main.py#L69)
- **證據等級：** A（已確認）

```python
'school_previously_infected_index_list': getattr(data, 'school_previousl', None),
#                                                       ↑ 屬性名打錯
```

`school_previousl` 屬性不存在，`getattr` 永遠回傳 None。**不影響模擬本身**，但所有儲存資料中學校層的 previously_infected 索引皆為 None，破壞依賴此欄位的事後分析。

**修正：**
```python
'school_previously_infected_index_list': getattr(data, 'school_previously_infected_index_list', None),
```

---

## 2. 參數設計缺陷（Bounds 設計讓 mean constraint 被繞過）

### 問題 5：Infectious period mean 上限被交叉參數繞過

- **位置：** [parameters_for_initialization.py:874-886](parameters_for_initialization.py#L874-L886)
- **證據等級：** A（參數值已驗證）

```python
infectious_period_shape = 4          # 固定初始值
infectious_period_mean_ub = 14
infectious_period_scale_ub = 14 / 4 = 3.5   # 用固定 shape=4 算 scale 上限
```

**問題：** shape 本身可優化（bounds 2–6），但 scale 的 ub 是用固定 shape=4 算的。當 Firefly 把 shape 推到 6：

```
max mean = shape_ub × scale_ub = 6 × 3.5 = 21 天   （設計意圖是 ≤ 14 天）
```

**目前最佳解：** shape=5.963, scale=3.486 → **mean = 20.79 天**，遠超 14 天上限。Firefly 利用超長感染期補償傳播機率不足。

**修正方向：** 改用對 mean 的直接約束，或讓 scale 上限隨 shape 連動；最簡單是將 shape 固定、僅優化 scale，或直接收緊 scale_ub。

---

### 問題 6：Incubation period 幾乎被 Truncation 架空

- **證據等級：** A（已驗證）

Incubation gamma 的 nominal mean = 0.755 × 1.664 = **1.26 天**，但 `draw_incubation_period()` 要求 `lower_bound = latent_period`（mean ≈ 3.77 天）。

```
P(gamma(0.755, 1.664) > 4) = 0.055
```

僅 5.5% 機率質量在下限以上 → 幾乎所有 incubation 樣本被截斷到緊鄰 latent period。後果：

- **前驅症狀傳播視窗 ≈ 0 天**（incubation − latent ≈ 0），消除了 COVID-19 已知的症狀前傳播。
- Incubation 的 Gamma 參數幾乎無作用，在此範圍優化無意義。

**修正方向：** 重新檢視 incubation 的 Gamma 參數與 bounds，使 nominal mean > latent mean，恢復合理的症狀前窗口。

---

### 問題 7：Attack rate 時間重映射在 incubation ≈ latent 時崩潰

- **位置：** [Data_synthesize.py:833-841](Data_synthesize.py#L833-L841)
- **證據等級：** A（已驗證）

```python
attack_rate_index = np.round(np.linspace(0, 14, incubation_period-latent_period+1))
# incubation=4, latent=4 → linspace(0,14,1) = [0]，只用 1 個點
attack_rate_index = np.append(attack_rate_index,
    np.round(np.linspace(15, 24, latent_period+infectious_period-incubation_period)))
# linspace(15,24,21) → 21 個點
```

25 個 attack rate 值中，indices 1–14（pre-symptom zone）幾乎從未被使用，所有感染期映射到 indices 15–24。Firefly 優化前半段的值是多餘維度，也部分解釋 attack-rate curve 的 irregular 形狀。

**註：** 此問題與問題 6 同源——incubation 被壓到接近 latent，導致 pre-symptom 區段塌縮。修好問題 6 後此問題大幅緩解。

---

### 問題 8：Symptomatic-to-ICU 比率偏高

- **證據等級：** A（參數值確認）

`symptom_to_recovered_transition_p` = 0.5845 → **41.6% 有症狀者進 ICU**。初始設定 0.82（台灣 ~18% 重症率）被 Firefly 推到下限 0.574（≈42.6% ICU）。ICU 轉入率被高估約一倍以上，影響 date_of_critically_ill 分布與感染窗口計算。

**修正方向：** 檢視 `symptom_to_recovered_transition_p` 的 bounds 來源；下限 0.574 是否反映真實數據需確認。

---

## 3. 模型邏輯異常（非 code 錯誤，但生物學不合理）

| # | 問題 | 證據 | 摘要 |
|---|---|---|---|
| 9 | Age risk ratios 全 < 1 | A | Best-fit [0.030, 0.129, 0.910, 0.542] 全 < 1，高風險的 40-59（文獻 RR=2.19）、60+ 也被壓制。方向與文獻相反。 |
| 10 | School `symptom_p > healthy_p` | A | healthy_p=0.0153, symptom_p=0.0426（2.8倍）。學生生病後接觸率「上升」，與直覺相反。尖峰在 τ≈+10 天、落在隔離邊界多被截斷，但邏輯不合理。 |
| 11 | HealthCare `symptom_phase`=9.36 天 | A | 就醫高峰在症狀後 9-14 天，但台灣早期確診多在症狀後 2-5 天，時機偏移。 |
| 12 | 傳播窗口延伸至隔離後 14.5 天 | A | Infectious mean 20.79 天 vs Monitor isolation 10.10 天。Code 正確填 0，但大部分感染期落在隔離後（無效），infectious period 幾乎不影響傳播。與問題 5 同源。 |
| 13 | Critically-ill→recovered mean = 43.84 天 | A | ICU 住院 43 天遠高於文獻（14-21 天）。從台灣小樣本 fit，需查異常值。 |
| 14 | Asymptomatic→recovered mean = 41.1 天 | A | 無症狀恢復 41 天極不尋常（文獻 14-20 天）。僅 2 個數據點 fit，樣本量不足。 |

問題 10、11 須確認尖峰是否在 monitor_isolation 截斷前造成實際傳播影響（部分為等級 C）。

---

## 4. 尚待確認的潛在問題（證據等級 C，勿直接改）

| 代號 | 位置 | 待驗證問題 |
|---|---|---|
| A | [Data_synthesize.py:596-615](Data_synthesize.py#L596-L615) | `generate_first_contact_matrix` Phase 1（sequential scan）與 Phase 2（proportional sampling）產生不同的 first-contact-day 分布，混合後統計不均勻。Claude3.5 重構引入。 |
| B | [Data_synthesize.py:326-331](Data_synthesize.py#L326-L331) | Disease course rejection sampling loop 在某些參數組合下 acceptance rate 可能極低，需量測是否為模擬慢的原因。 |
| C | — | `vaccination_rate`=0 鎖死，`vaccine_efficacy`(P[69]=0.899) 從不影響輸出，但仍佔 198 維搜索空間，浪費收斂資源。 |
| D | — | Municipality `healthy_p`(0.0107) < `symptom_p`(0.0493)，社區接觸生病後上升；尖峰 τ=7-17 天可能已隔離，需確認實際影響。 |

---

## 5. 你原先未提到、建議額外確認的面向

| 面向 | 待確認問題 |
|---|---|
| **Scoring function 對稱性** | Cost function 同時優化 contact array 與 attack rate（scale 不同）。normalized 後兩者相對權重是否合理？ |
| **Energy distance 穩定性** | [firefly_optimizer.py:415](firefly_optimizer.py#L415) `estat()` 用 nboot=100，每次評估 energy_cost 高隨機性，可能讓 Firefly 目標函數 noisy、難收斂。 |
| **接觸 vs 感染率解耦** | 160 個 attack rate 參數 + 35 個接觸參數可相互補償（高接觸×低 attack rate ≡ 低接觸×高 attack rate），造成大量 local optima。 |
| **latent < incubation 約束** | rejection sampling 僅確保 `latent ≤ monitor_isolation`，未明確確保 `latent < incubation`。目前因 truncation 數值上等價，但語意約束未在 code 表達。 |
| **負 loc 的 gamma** | `symptomatic_to_critically_ill_loc` lb=-0.6064，best=-0.113。負 loc 技術上可行，需確認 truncated sampling 無 numerical issue。 |
| **Firefly alpha decay** | `new_alpha *= 0.97`，200 代後 alpha≈0.0022。需看 convergence curve 確認是否過早收斂。 |

---

## 6. 優先級總結與建議執行順序

> 優先序已依「證據強度 + 是否需先核對 Supplementary」修正，與初版報告不同。

| 順序 | 項目 | 證據 | 前置條件 | 影響層面 |
|---|---|---|---|---|
| **0** | 取得論文 Supplementary，核對 S11（overdispersion 公式、vaccination 機制、overdispersion 粒度） | — | — | 決定 Bug 1/2/3 是否成立 |
| **1** | Bug 1: overdispersion `**` → `*` | A+B | 核對 S11 為乘法 | 模型行為與 Supplementary 一致 |
| **2** | 問題 5: infectious period mean 上限設計（20.79 vs 14 天） | A | — | Firefly 不再濫用超長感染期 |
| **3** | 問題 6+7: incubation truncation 與 attack-rate 映射塌縮 | A | — | 恢復症狀前傳播、釋放冗餘維度 |
| **4** | Bug 2: draw_vaccination_status 用錯 rate | A+B | 核對 S11 vaccination 意圖 | 決定是否移除 16.6% 免疫抑制 |
| **5** | Bug 3: overdispersion per-day vs per-person | A+B | 先修 Bug 1 再評估 | 隨機結構 |
| **6** | 問題 8: 42% symptomatic→ICU | A | — | Disease course 分布 |
| **7** | 問題 9: age risk ratios 全 < 1 | A | re-optimization | 年齡風險方向 |
| **8** | Bug 4: school 拼字錯誤 | A | — | 僅事後分析資料 |
| **9** | 問題 10-14 + 潛在問題 A-D + §5 面向 | A/C | re-optimization 後評估 | 逐項驗證 |

**核心原則：**
1. **先核對 Supplementary，再動 code。** 所有等級 B 的結論在核對前不可視為定案。
2. **改 bounds 必須重跑 Firefly。** 否則 best-fit 參數可能落在新 bounds 之外，結果無意義。
3. **不可一次套用所有修正。** 各參數交互作用非線性（診斷報告 Finding 10：四項全改 R0 proxy 衝到 19）。採 constrained re-optimization，逐步驗證。
