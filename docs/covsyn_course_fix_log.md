# CovSyn 疾病進程修正紀錄（Course / Penalty Fix Log）

> 本文件記錄 2026-06-29 對 CovSyn 的一批修正:**用「軟性懲罰(soft penalty)」把疾病進程時間、年齡風險、重症率推向文獻值**,而**不是把參數鎖死**。任何人照本文件都能理解改了什麼、為什麼、以及如何還原。

---

## 1. 為什麼要改(驗證發現)

依 `covsyn_validation_workflow.md` 對「優化後參數」抽 10,000 條 disease course 驗證,發現幾個明顯偏離文獻的值:

| 指標 | 修正前 | 文獻 | 問題 |
|---|---|---|---|
| infectious period | mean 13.5 d(max 49) | 7-10 d | 過長,1.2% 超過 30 天 |
| 無症狀 infection→recovery | mean 62 d | 14-20 d | 嚴重過長(原本只用 2 筆資料 fit) |
| 症狀 onset→recovery | mean 33 d | 14-20 d | 過長 |
| ICU→recovery | mean 36 d | 14-21 d | 過長 |
| symptom→ICU 比率 | 40% | ~18%(台灣) | 過高 |
| age 40-59 / 60+ 風險比 | 0.80 / 0.45 | 2.19 / 1.75(Cheng) | 方向相反(高風險組被壓低) |
| latent / incubation | 2.6 / 4.7 d | 3-4 / 5-6 d | 略低 |

> Stage 1(不可能的事件順序,如 confirmed 早於 infection 等)**全數通過**,故問題在「分布數值」而非「邏輯」。

---

## 2. 修正哲學:懲罰,而非鎖死

兩種做法的差別:

- **鎖死(lock)**:把參數的上下界設成同一個值 → 參數固定、不再優化。簡單但失去彈性。
- **軟性懲罰(本次採用)**:參數**仍可自由優化**,但只要它對應的「疾病進程平均天數 / 年齡比 / 重症率」**落在文獻範圍內,懲罰 = 0(完全不影響擬合)**;一旦**超出範圍,懲罰隨偏離程度平方成長**,把它拉回來。

好處:模型仍能在「合理的文獻範圍內」自行找最佳擬合,只是不准跑到不合生理的數值。

---

## 3. 具體改了哪些檔

### (A) `firefly_optimizer.py` — 加入 `physiology_penalty`

在 `cost_function` 之前新增一個函式 `physiology_penalty(P)`,並把它加進目標函數:

```python
total_cost = sum(costs) + energy_weight * energy_cost + physiology_penalty(P)
```

懲罰的計算方式(相對距離、平方;範圍內為 0):

| 受罰量 | 由哪些參數算出 | 文獻目標範圍 |
|---|---|---|
| latent mean | P[37]·P[38] | 3-4 d |
| incubation mean | P[41]·P[42] | 5-6 d |
| infectious mean | P[39]·P[40] | 7-10 d |
| 無症狀→recovery mean | P[46]·P[47]+P[48] | 14-20 d |
| 症狀→recovery mean | P[52]·P[53]+P[54] | 14-20 d |
| ICU→recovery mean | P[55]·P[56]+P[57] | 14-21 d |
| symptom→ICU 比率 | P[196] | 目標 0.82(=18% ICU) |
| age 風險比 ×4 | P[63..66] | Cheng 0.3 / 1 / 2.19 / 1.75 |

- 權重:`PHYSIOLOGY_PENALTY_WEIGHT = 2.0`(可調)。
- 數值檢核:在「修正前的最佳參數」上 penalty = **12.58**(高,代表偏離);在「文獻理想參數」上 penalty = **0.0**。

### (B) `variable/course_parameters{,_lb,_ub}.npy` — 只放寬 3 條 recovery gamma 的 bounds

懲罰只能在「參數界線(bounds)允許的範圍內」把值拉動。三條 recovery 分布原本的下界太高,**文獻值(14-21 天)根本到不了**,所以必須先放寬下界(不是鎖死,是擴大可行範圍):

| 參數(course index) | 修正前 bounds | 修正後 bounds |
|---|---|---|
| 無症狀→rec shape / scale / loc (9,10,11) | [4.6,6.9] / [7.2,10.8] / [0,0] | [3,9] / [1.5,4] / [0,3] |
| 症狀→rec shape / scale / loc (15,16,17) | [1.7,2.5] / [8.4,12.6] / [4.4,6.0] | [3,9] / [1.5,4] / [0,3] |
| ICU→rec shape / scale / loc (18,19,20) | [1.2,1.8] / [14.1,21.2] / [7.4,11.0] | [3,9] / [1.5,4] / [0,6] |

> infectious / age / ICU 比率的 bounds **本來就涵蓋文獻值**,故不動 bounds,只靠懲罰拉動。
> 20-39 age 風險比先前已固定為 1(= Cheng 參考值),維持不變。

---

## 4. 沒有處理的項目(已知、刻意保留)

| 項目 | 原因 |
|---|---|
| municipality 候選接觸機率 ≈ 0(7e-6) | 屬 firefly 優化結果,可能為設計意圖(社區=大量低機率接觸);需更深入討論再決定是否加約束。 |
| scoring 各項尺度 / energy nboot=1 噪音 | 屬優化器穩健性議題,與本次「疾病進程數值」不同層次。 |

---

## 5. 影響與後續

- **必須重跑 firefly**:懲罰只在優化過程作用;改完要重新訓練,best-fit 才會移到文獻範圍內。
- 重跑後會**重新產生 synthetic data 並再次驗證**(Stage 1/2 應全部落在文獻範圍)。
- 疾病進程的 **distribution 形狀(變異數)** 本次只約束「平均值」,未約束變異;若日後發現變異不合理,再加對應懲罰。
- 把 infectious period 變短,會讓 `latent+infectious ≥ monitor_isolation` 的拒絕抽樣稍微變嚴,需留意 disease course 的 acceptance rate(目前 test 跑正常)。

---

## 6. 如何還原(reverse)

所有被改的檔在修改前都已備份:

```
firefly_optimizer.py.bak_20260629          # 還原 penalty 修改
variable/course_parameters.npy.bak_widen_20260629
variable/course_parameters_lb.npy.bak_widen_20260629
variable/course_parameters_ub.npy.bak_widen_20260629
```

還原指令(遠端):

```bash
cp firefly_optimizer.py.bak_20260629 firefly_optimizer.py
cp variable/course_parameters.npy.bak_widen_20260629    variable/course_parameters.npy
cp variable/course_parameters_lb.npy.bak_widen_20260629 variable/course_parameters_lb.npy
cp variable/course_parameters_ub.npy.bak_widen_20260629 variable/course_parameters_ub.npy
```

---

## 7. 套用這批修正所用的腳本(留存)

| 腳本 | 作用 |
|---|---|
| `patch_firefly_penalty.py` | 在 `firefly_optimizer.py` 插入 `physiology_penalty` 並接進目標函數(idempotent,先還原再套用) |
| `widen_recovery_bounds.py` | 放寬 3 條 recovery gamma 的 .npy bounds(先備份) |
| `validate_full.py` | 抽 10,000 條 course 做 Stage 1+2 驗證、印出 Stage 3/4 參數結構(改前/改後對照用) |

---

**結論:** 本次以「軟性懲罰 + 放寬必要 bounds」把疾病進程時間、年齡風險、重症率約束到文獻範圍,參數仍可優化、範圍內不受干擾。改完已用 `--mode test` 驗證可正常執行,接著進行正式重跑與重新驗證。
