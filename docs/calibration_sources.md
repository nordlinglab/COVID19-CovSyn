---
title: Sources of every CovSyn calibration value
author: Rex Lee
date: 2026-10-15
purpose: Lists every value that calibrates or constrains CovSyn, where it is used, the source the code claims, and whether that source is stored in lit/ with the supporting sentence or table quoted, so that no value rests on our own judgement.
---

# CovSyn 校正數值來源總表

**規則（使用者 2026-10-15）**：每個數值都要有明確來源。
來源是論文時，論文要存進 `lit/`（依 `.agents/guides/lit-sources.md`），並在本表「原文證據」欄摘錄支持該數值的字句或圖表、標明頁碼／表號。
來源是資料時，要寫出資料檔與計算程式。
沒有來源的值必須標為「假設」。

**2026-10-15 採用（B61，見 `covsyn_decisions.md`）**：1.2 家戶 [2.27, 3.28]、1.3 學校 [1.15, 1.72]、1.8 採 12.6% 定義（58/460）、3.12 冪律指數 1.67、4.8 三項都檢查（推算的感染→隔離 [8, 14]、台灣直接量的感染→確診 [11.4, 14.6]、潛伏期對台灣 [4.96, 7.57] 與文獻 [3.9, 8.0]）、新增社區與職場單日分布對 Fu 2010（只量）、k 改用 MLE。
未採用、待後續：3.8–3.10 與 4.13–4.14（職場重新設計，待工商普查行業別表）；city term 不需改程式（B27 起已不影響接觸數）。

**證據狀態**

- ✅ 論文已存、原文已摘錄、數值已對照
- 🟡 論文已存，但原文尚未摘錄或尚未對照
- ⬜ 論文未存，或尚未查證
- ❗ 已查證，來源不支持此值
- 📐 由資料計算（需列資料檔與程式）
- 🔧 假設或調參，沒有外部來源

程式位置縮寫：`FO` = `calibration/firefly_optimizer.py`，`SA` = `calibration/sar_anchors.py`，`AP` = `calibration/apply_phase_d_parameters.py`，`DS` = `model/data_synthesize.py`，`VP` = `validation/verify_phase_d.py`。

---

## 1. 目標函數的校準目標（`FO` `OUTCOME_TARGETS`，共 29 項）

| # | 量 | 區間 | 程式宣稱的來源 | 狀態 | 原文證據／計算 | 待辦 | 建議新值 |
|---|---|---|---|---|---|---|---|
| 1.1 | offspring_k | 0.20–0.35 | 台灣疫調 NegBin MLE k = 0.29 | 📐 | 資料：Wu & Nordling 2024 疫調資料集；程式待指認 | 找出計算 0.29 的程式與輸入 | 重算 MLE 0.286（上界定義，n = 579）；下界定義 0.108。維持 0.20–0.35。⚠ `verify_phase_d` 的 k 用動差法（0.262），與目標的 MLE 不同，應統一 |
| 1.2 | daily_household | 1.50–2.20 | 2020 全國調查 6.03 × Fu 2012 場所比例 | ❗ | 兩篇已存（`2026-10-15-a9f5a2a6`、`2026-10-07-4c8c5035`）；2020 調查 Table 4：家戶 2.27 ± 0.03、非家戶 4.28 ± 0.06；與相乘值 1.86 不一致 | 重新決定目標的來源與算法 | Fu 2010 ≥15 分鐘：3.28（全部接觸 3.83）；2020 調查 2.27。建議 [2.27, 3.28]：疫情期調查為下界、疫情前全國樣本為上界（待決定） |
| 1.3 | daily_school | 0.90–1.40 | 同上 | ⬜ | | 同上 | Fu 2010 ≥15 分鐘：1.72（全部接觸 2.36）。現行區間偏低；建議依同一定義改為約 1.7（待決定） |
| 1.4 | daily_workplace | 1.20–1.80 | 同上 | ❗ | 2020 調查每人最多記 12 位接觸、問「昨天」（含週末）；職場目標可能偏低 | 職場重新設計時一併處理 | Fu 2010 ≥15 分鐘：1.65（全部接觸 2.99；平日 1.92）。落在現行區間內；建議保留區間，來源改為 Fu 2010 |
| 1.5 | daily_municipality | 1.20–1.80 | 同上 | ⬜ | | 同上 | Fu 2010 ≥15 分鐘：1.76（全部接觸 3.23；平日 1.49）。落在現行區間內（近上限）；建議保留區間，來源改為 Fu 2010 |
| 1.6 | daily_health_care | 0–0.30 | 假設（健康者幾乎不接觸醫護） | 🔧 | | 註明為假設 | — |
| 1.7 | asymptomatic_share | 0.20–0.28 | 台灣疫調：無發病日 137/579 = 23.7%；嚴格定義 119/579 = 20.6%（E47） | 📐 | | 列出計算程式 | 重算 20.4%（嚴格，118/578）至 23.5%（寬鬆，136/578）。維持 [0.20, 0.28] |
| 1.8 | icu_share_of_symptom | 0.06–0.20（只量） | 台灣疫調 56/442 = 12.7%（B33） | 📐 | | 同上 | 重算：要求發病日 9.05%（40/442）；ICU 視為有症狀 12.6%（58/460）。舊值 56/442 的分子含無發病日的 ICU 病例、分母不含，需選定定義；現行區間兩者都涵蓋 |
| 1.9 | death_share_of_icu | 0.08–0.18（只量） | 台灣疫調 7/56 = 12.5% | 📐 | | 同上 | 重算 12.1%（7/58）。維持 |
| 1.10 | case_fatality | 0.008–0.018（只量） | 台灣疫調 7/579 = 1.2% | 📐 | | 同上 | 重算 1.21%（7/578）。維持 |
| 1.11 | sar_household | 4.6–10.1% | Cheng 2020 每例感染 10/100 ÷ CovSyn 候選接觸數（`SA`，E55） | 🟡 | `lit/markdown/cheng2020_TracingSymptomJAMAInternMed.md` 已存；Table 2 數字待摘錄 | 摘錄 Table 2 | — |
| 1.12 | sar_school | 1–4% | 「出處本身尚未釐清」（`SA` 註解，B4） | ❗ | | 找出處，找不到就改 | — |
| 1.13 | sar_workplace | 1.5–5%，中心 3.4% | 同上；3.4% 等於 Chen 2022 英格蘭職場攻擊率中位數，但未記錄 | ❗ | | 同上 | — |
| 1.14 | sar_health_care | 0.1–1.6% | Cheng 2020 6/100 ÷ CovSyn 候選接觸數 | 🟡 | 同 1.11 | 同 1.11 | — |
| 1.15 | sar_municipality | 0.1–1.0% | Cheng 2020 1/100；下限 0.1% 為設定 | 🟡／🔧 | 同 1.11 | 同 1.11 | — |
| 1.16 | medical_late_share | 25–50% | Cheng 2020 醫療接觸：發病 8 天後 256/697 = 36.7% | 🟡 | 同 1.11 | 同 1.11 | — |
| 1.17 | medical_early_share | 40–70% | Cheng 2020：發病前 33.9% + 0–3 天 21.5% = 55.4% | 🟡 | 同 1.11 | 同 1.11 | — |
| 1.18 | community_median | 3–15 | 台灣疫調 38 筆（friend + other） | 📐 | `validation_reference/taiwan_tracing_contacts_per_case.csv` | 列出程式；M3 要求改為 community-only | 重算 7.5（bootstrap 95% 3.5–32.5，n = 38，每例整段追蹤期）。Fu 2010 單日：中位數 2（≥15 分鐘為 0）。維持 [3, 15]，須註明是每例累積 |
| 1.19 | community_p90 | 20–200（只量） | 同上 | 📐 | | 同上 | 重算 195（68–822）。只量，維持 |
| 1.20 | community_tail_ratio | 5.5–93.1 | 同上，bootstrap 95% CI（E77、B50） | 📐 | | 同上 | 重算 26.0（5.5–93.9，20,000 次）。與現行一致 |
| 1.21 | community_zero_share | 0–0.30（只量） | 未記錄 | ⬜ | | 找依據 | 疫調資料無法計算（38 筆依定義皆非零）。Fu 2010 單日：40.7%（≥15 分鐘 62.0%）。建議以 Fu 2010 為來源，需換算成每例累積 |
| 1.22 | incubation_mean | 3.9–8.0 天 | 文獻報告平均值範圍 | ⬜ | E45：清單在 `plot_result/plot_synthetic_data.ipynb` 的 `report_*` 字典（15 篇；3.9 來自 Ki） | 列出每個端點的論文、存檔並摘錄原文 | — |
| 1.23 | pre_onset_window_mean | 1–3 天 | 未記錄（Cheng 2020「傳播集中在發病前後」） | ⬜ | | 找依據 | — |
| 1.24 | pre_onset_zero_share | 0–12% | 未記錄 | ⬜ | | 找依據或標為假設 | — |
| 1.25 | closure_after_confirmation_symptomatic | 22–32 天 | 台灣疫調 確診→解除隔離 平均 27.0（n = 56，E91） | 📐 | | 列出程式 | 重算 27.0（22.1–32.2，n = 56，含 ICU 病例）。一致 |
| 1.26 | closure_after_confirmation_asymptomatic | 17–31 天 | 台灣疫調 平均 23.3（n = 23） | 📐 | | 同上 | 重算 23.3（16.9–31.0，n = 23）。一致 |
| 1.27 | icu_to_closure | 29–44 天 | 台灣疫調 平均 36.2（n = 28，E93） | 📐 | | 同上 | 重算 36.2（29.0–44.1，n = 28）。一致 |
| 1.28 | onset_to_icu | 5.4–9.2 天 | 台灣疫調 平均 7.2（n = 40，E92） | 📐 | | 同上 | 重算 7.2（5.4–9.2，n = 40）。一致 |
| 1.29 | onset_to_confirmation | 5–7 天（中位數） | 台灣疫調 中位數 6（n = 442）；Ge 2021 發病→隔離 5 天 | 📐／🟡 | `lit/markdown/ge2021_CohortZhejiangJAMAInternMed.md` 已存 | 摘錄 Ge 2021 原文 | 重算中位數 6（n = 442），平均 7.9（7.3–8.6）。一致 |

## 2. 目標函數的其他部分

| # | 項目 | 數值 | 程式宣稱的來源 | 狀態 | 待辦 | 建議新值 |
|---|---|---|---|---|---|---|
| 2.1 | Cheng 2020 接觸分箱擬合（`cost_contact`） | Table 2 各層各時段接觸數與攻擊率 | Cheng 2020 | 🟡 | 摘錄 Table 2 全表 | — |
| 2.2 | 與台灣個案資料的能量距離 | — | Wu & Nordling 2024 資料集 | 📐 | 列出資料檔 | 資料檔：`figshare_taiwan_covid.xlsx`（578 列，缺案 530） |
| 2.3 | latent 平均 | 4.1–5.5 天（`physiology_penalty`） | 4.1 = Cheng 2020 中位數；5.5 = Xin 2022 平均（E45：兩端統計量不同） | 🟡 | 兩篇都已存；摘錄原文；處理「中位數 vs 平均」混用 | — |
| 2.4 | infectious 平均 | 3.45–20 天 | 文獻報告平均值範圍 | ⬜ | 列出端點論文 | — |
| 2.5 | pre-onset 窗口（參數） | 1–3 天 | 同 1.23 | ⬜ | 同 1.23 | — |
| 2.6 | onset→confirmation（參數） | 1–12 天 | 原論文模型上限 12.02 天 | 🟡 | 對照 Wu 2023 原文 | — |

## 3. 固定常數與鎖定參數

| # | 常數 | 值 | 程式位置 | 程式宣稱的來源 | 狀態 | 待辦 | 建議新值 |
|---|---|---|---|---|---|---|---|
| 3.1 | `TRACING_DELAY_MEAN` | 1 天（Poisson） | `DS` | PMC7605820：台灣疾管署約一天內匡列 | ⬜ | 存論文、摘錄原文；M6 要求改成可調參數 | — |
| 3.2 | 疫調涵蓋率 | 100%（感染源確診後，所有被他感染的人都會被追到） | `DS` 第 716–731 行 | 未記錄（隱含假設） | 🔧 | 標為假設；找文獻 | — |
| 3.3 | `HEALTH_CARE_POST_ISOLATION_DAYS` | 14 天 | `DS` | 程式註解：假設，無台灣來源 | 🔧 | | — |
| 3.4 | `HEALTH_CARE_POST_ISOLATION_PROTECTION` | 0.25 | `DS` | 同上 | 🔧 | | — |
| 3.5 | `SYMPTOM_TO_ICU_AGE_RR` | 0.05、1、2.8、6.4 | `DS` | Verity 2020 Table 1（有症狀→住院比例） | ⬜ | 存論文、對照 Table 1 推算 | — |
| 3.6 | `ICU_TO_DEATH_AGE_RR` | 0.10、1、2.3、12.8 | `DS` | Verity 2020 Table 1（IFR ÷ 住院） | ⬜ | 同上 | — |
| 3.7 | `AGE_RISK_RATIOS`（感受性） | 0.39、1、1.90、1.44 | `AP` | B2／B14；來源待查 | ⬜ | 找出處 | — |
| 3.8 | `WORK_GROUP_EXPONENT` | 0.27 | `DS` | Chen 2022 Table 5 | ❗ | 2026-10-12 查證：Table 5 依行業，16→650 與 3→8 不是同一列；逐列迴歸指數約 0.06。2026-10-15：沒有找到任何依公司規模的職場接觸研究；文獻模型用固定約 20 人的工作群組，見 [`source_quotes.md`](source_quotes.md) §2 | 移除。Timpka 2016：群組由 1 人增至 50 人以上，每人身體接觸只從約 1 增至 2。群組大小改依工商普查行業別（Epicast 做法，上限 86），固定 20 人做敏感度分析（Timpka、FluTE） |
| 3.9 | `WORK_GROUP_COEFFICIENT` | 3.0 | `DS` | 對齊台灣疫調同事接觸中位數 2–2.5（n = 6） | 📐／❗ | 樣本極小；職場重新設計 | 重算疫調同事接觸中位數 2（n = 8，不是 6）。Fu 2010 平日受僱者每天見面的職場接觸：中位數 2、平均 4.79；建議以 Fu 2010 為驗證目標 |
| 3.10 | `WORK_GROUP_LOG_SIGMA` | 0.6 | `DS` | 未記錄 | 🔧 | | 職場重新設計後決定 |
| 3.11 | 大型事件機率下限 P[199] | 0.05 | `AP`（B59） | 探測：0–0.05 期望成本無差異 | 🔧 | M3 要求改成每人直接抽 power-law，此項可能取消 | — |
| 3.12 | 大型事件冪律指數 P[200] | 1.49 | `AP` | 38 筆尾部 MLE（Clauset 2009） | ❗ | 2026-10-12 重算得 1.67（KS 0.081–0.113），1.49 無法重現 | 重算 1.668（x_min 21，尾巴 17 筆），重現 1.67；程式用的 1.49 應改為 1.67。參考：Danon 2012 為 2.45（英國，含群體接觸） |
| 3.13 | 大型事件規模 21–1000 | 21、1000 | `AP` | 21 = KS 選出的 k_min（已重現）；1000 = 上限設定 | 📐／🔧 | | 重算 x_min = 21。一致 |
| 3.14 | 事件接觸攻擊率倍數 P[203] | 1 | `AP` | 教授 2026-09-29「先只改接觸數分布」 | 🔧 | | — |
| 3.15 | 傳染力倍數 Gamma 形狀範圍 | 0.15–0.8 | `AP` | 調參（避免退化） | 🔧 | | — |
| 3.16 | 社區接觸計數離散度 P[198] | 0.05–1.0 | `AP` | 調參 | 🔧 | | — |
| 3.17 | 依發病日的攻擊率形狀 | 25 個時段 | `SA` | Ge 2021 Figure 2（調整後相對風險） | 🟡 | 已存；摘錄 Fig 2 與原文三個數值 | — |
| 3.18 | 各層接觸機率範圍（`CONTACT_BOUNDS`） | 見程式 | `AP` | 調參，依每日接觸目標推算 | 🔧 | | — |
| 3.19 | 嚴重度參數範圍 P[195]–P[197] | 見程式 | `AP` | 台灣疫調（同 1.7–1.9） | 📐 | | 同 1.7–1.9 的重算值 |

## 4. 只用於驗收的參考值（`VP`）

| # | 量 | 區間 | 宣稱來源 | 狀態 | 待辦 | 建議新值 |
|---|---|---|---|---|---|---|
| 4.1 | 家戶其他成員、獨居 | 2.6–3.0、11–16% | 內政部 2021 戶籍 | ⬜ | 存資料表與出處 | — |
| 4.2 | 各學制班級人數 | 見清單 | 教育部學生數 | ⬜ | 同上 | — |
| 4.3 | 公司規模 | — | 2016 工商普查 Table 18、18-2 | ⬜ | 同上；範圍與單位問題見 B4 | 110 年工商普查：1,549,583 個場所、9,605,372 人（平均約 6.2 人，我的推算）；行業別表待下載 |
| 4.4 | Jian 2020 追蹤窗口接觸數 | 13.9–19.1 | Jian 2020 每例 16.5 | 🟡 | Markdown 已存（`jian2020_TracingCOVIDIntJInfectDis.md`），PDF 在 Downloads；摘錄原文 | — |
| 4.5 | 0–19 歲年齡風險比 | 0.34–0.77 | Zhang 2020 0.34、Viner 2021 0.56、Uthman 2024 0.58、Madewell 2020 0.59 | 🟡／⬜ | Zhang 已存；其他三篇未存；0.77 上限出處待查 | — |
| 4.6 | 40–59、60+ 年齡風險比 | 1.63–2.03、1.12–1.52 | Cheng 2020 1.83、1.32 | 🟡 | 摘錄原文 | — |
| 4.7 | R、感染 3 人以上、最多感染數 | 0.047–0.43、0.2–4.3%、5–30 | 台灣疫調兩種定義（E80） | 📐 | | 重算：上界 R 0.428、≥3 人 4.32%、最多 8；下界 R 0.047、0.17%、3。一致 |
| 4.8 | 感染→隔離 | 8–14 天 | 潛伏期約 5.3 天 + 發病→確診 5–7 天（推算） | 📐／⬜ | 5.3 天的出處待查 | 直接由資料算：感染→確診平均 12.9（11.4–14.6，n = 33），中位數 13；潛伏期平均 6.2（n = 28）。建議改用直接量，區間約 [11, 15] |
| 4.9 | 家人每天被接觸的比例 | ≥ 50% | Fu 2012 | 🟡 | 已存（`2026-10-07-4c8c5035`）；待摘錄支持 50% 的原文 | — |
| 4.10 | 縣市效應置換檢定 | p ≥ 0.05 | 設計要求（E4、E83） | 🔧 | 2026-10-15 文獻：面對面調查多為無差異或方向不一；手機資料每人指數約 0.12。原文見 [`source_quotes.md`](source_quotes.md) §1；處理方式待決定 | ISSP 2017 與 Fu 2010 都不支持城市效應（`source_quotes.md` §3.3） |
| 4.11 | ICU 比例 60+ ÷ 20–39 | 3–12 | Verity 2020 梯度 | ⬜ | 同 3.5 | — |
| 4.12 | 台灣首波情境平均年齡 | 33–42 | 28 個源頭病例年齡 | 📐 | | 重算 37.6（組中點，36.3–38.9，n = 577）；舊 37.1 為組內依人口分配。維持 [33, 42] |
| 4.13 | 職場：大公司／小公司每例感染比 | 1–3 | Chen 2022「2.7 倍」 | ❗ | 誤讀，見 3.8；論文已存（`2026-10-07-74078dfd`） | 移除（Chen 2022 誤讀） |
| 4.14 | 職場：每員工感染比 | 4–60 | Chen 2022 17.8% ÷ 1.1% = 16 倍 | 🟡 | 原文已核對（攻擊率中位數依企業規模）；論文已存（`2026-10-07-74078dfd`），待把原文摘入 `source_quotes.md` | — |
| 4.15 | 病程時間文獻範圍（潛伏期、傳染期、世代時間、序列間隔等） | 見 `reproduce_wu2025.literature_ranges` | 原 notebook 的文獻表 | ⬜ | 逐一列出論文（與 M10 的 SI／GT 文獻表一起做） | — |

## 5. 各情境每日接觸：Fu 2010 台灣全國接觸日誌

資料：`2026-10-15-467602c6`（Zenodo 10.5281/zenodo.6385759，CC BY 4.0），存於 `covid19-research/data/external/fu2010_taiwan_contact_zenodo6385759/`。
程式：`covid19-research/scripts/summarize_taiwan_contact_data.py`（`by_setting`）。
每人每日平均；一筆接觸若標了多個地點（24,265 筆中 947 筆），依疫調擷取的優先順序（家 > 學校 > 工作 > 社區）只算一次。
「≥15 分鐘」是 `duration_multi` 代碼 3–5（15–59 分鐘、1–4 小時、4 小時以上），對齊 2020 調查與疫調的密切接觸定義；佔全部接觸的 67.6%。

| 母體 | 定義 | 家戶 | 學校 | 職場 | 社區 | 合計 |
|---|---|---|---|---|---|---|
| 全部日（n = 1,954） | 全部接觸 | 3.83 | 2.36 | 2.99 | 3.23 | 12.42 |
| 全部日（n = 1,954） | ≥15 分鐘 | 3.28 | 1.72 | 1.65 | 1.76 | 8.40 |
| 平日（n = 1,343） | 全部接觸 | 3.75 | 2.94 | 3.42 | 2.86 | 12.97 |
| 平日（n = 1,343） | ≥15 分鐘 | 3.20 | 2.10 | 1.92 | 1.49 | 8.72 |

≥15 分鐘的社區接觸：平均 1.76、0 次 62.0%、P90 6、P99 17、最大 37。
限制：2010 年（疫情前）；每人最多記 40 筆（3.9% 觸頂）；2020 調查（疫情期、每人最多 12 位）的家戶為 2.27、非家戶 4.28。

## 6. 由資料計算的數值：重新計算結果（2026-10-15）

程式：`src/covsyn/data_processing/recompute_taiwan_targets.py`（在 CovSyn 根目錄執行 `python -m covsyn.data_processing.recompute_taiwan_targets`），輸出 `validation_reference/recomputed_targets.csv`。
區間是 bootstrap 95%（20,000 次、種子 0）。
個案資料是 Wu & Nordling 資料集（`wu2024DatasetTracingSciData`）的公開版 `figshare_taiwan_covid.xlsx`（578 列，缺案 530）；疫調接觸的三個 CSV 由 `extract_tracing_reference.py` 與 `extract_coworker_contacts.py` 從 forecasting repo 的活頁簿擷取，該活頁簿不在本機，所以擷取這一步本次沒有重跑。

| # | 量 | 值 | n | 95% 區間 | 定義與算法 |
|---|---|---|---|---|---|
| 1.7 | asymptomatic share, broad (no onset date) | 0.2353 | 578 |  | cases without onset_of_symptom / all cases |
| 1.7 | asymptomatic share, strict (no onset date and symptom = X) | 0.2042 | 578 |  | cases with neither an onset date nor a symptom / all cases (E47) |
| 1.8 | ICU share of symptomatic, onset date required | 0.0905 | 442 |  | ICU cases with an onset date / cases with an onset date |
| 1.8 | ICU share of symptomatic, ICU counted as symptomatic | 0.1261 | 460 |  | all ICU cases / (cases with an onset date + ICU cases without one); an ICU case without an onset date is taken as a missing date, not as asymptomatic |
| 1.8 | ICU cases without an onset date | 18 | 578 |  | count, reported so the ICU numerator can be checked |
| 1.9 | death share of ICU | 0.1207 | 58 |  | ICU cases with a death date / ICU cases |
| 1.9 | deaths without an ICU date | 0 | 578 |  | count |
| 1.1 | case fatality | 0.01211 | 578 |  | cases with a death date / all cases |
| 1.25 | confirmation -> recovery, symptomatic, mean (days) | 20.19 | 36 | 15.81–25.06 | recovery - confirmed_date, cases with an onset date and no ICU; negative intervals dropped; interval = bootstrap 95% of the mean |
| 1.25 | confirmation -> recovery, symptomatic, median (days) | 16 | 36 |  | recovery - confirmed_date, cases with an onset date and no ICU |
| 1.25 | confirmation -> recovery, symptomatic incl. ICU, mean (days) | 26.96 | 56 | 22.05–32.18 | recovery - confirmed_date, cases with an onset date; negative intervals dropped; interval = bootstrap 95% of the mean |
| 1.25 | confirmation -> recovery, symptomatic incl. ICU, median (days) | 23 | 56 |  | recovery - confirmed_date, cases with an onset date |
| 1.26 | confirmation -> recovery, asymptomatic, mean (days) | 23.35 | 23 | 16.87–30.96 | recovery - confirmed_date, cases without an onset date; negative intervals dropped; interval = bootstrap 95% of the mean |
| 1.26 | confirmation -> recovery, asymptomatic, median (days) | 15 | 23 |  | recovery - confirmed_date, cases without an onset date |
| 1.27 | ICU -> recovery, mean (days) | 36.21 | 28 | 29–44.11 | recovery - icu; negative intervals dropped; interval = bootstrap 95% of the mean |
| 1.27 | ICU -> recovery, median (days) | 29.5 | 28 |  | recovery - icu |
| 1.28 | onset -> ICU, mean (days) | 7.2 | 40 | 5.35–9.225 | icu - onset_of_symptom; negative intervals dropped; interval = bootstrap 95% of the mean |
| 1.28 | onset -> ICU, median (days) | 5.5 | 40 |  | icu - onset_of_symptom |
| 1.29 | onset -> confirmation, mean (days) | 7.93 | 442 | 7.335–8.552 | confirmed_date - onset_of_symptom; negative intervals dropped; interval = bootstrap 95% of the mean |
| 1.29 | onset -> confirmation, median (days) | 6 | 442 |  | confirmed_date - onset_of_symptom |
| 4.8 | infection -> confirmation, mean (days) | 12.91 | 33 | 11.36–14.58 | confirmed_date - earliest_infection_date, cases with a recorded infection date; negative intervals dropped; interval = bootstrap 95% of the mean |
| 4.8 | infection -> confirmation, median (days) | 13 | 33 |  | confirmed_date - earliest_infection_date, cases with a recorded infection date |
| 4.8 | infection -> onset (incubation), mean (days) | 6.214 | 28 | 4.964–7.571 | onset_of_symptom - earliest_infection_date, cases with both dates; negative intervals dropped; interval = bootstrap 95% of the mean |
| 4.8 | infection -> onset (incubation), median (days) | 5.5 | 28 |  | onset_of_symptom - earliest_infection_date, cases with both dates |
| 4.12 | mean case age (band midpoints) | 37.56 | 577 | 36.25–38.9 | mean of the midpoints of the 10-year age bands; bands not of that form are left out (['nan']) |
| 1.1 | offspring k, upper definition, MLE | 0.2859 | 579 |  | infected contacts per case summed over layers, cases with no record = 0 (E80 upper definition) |
| 1.1 | offspring k, upper definition, method of moments | 0.2623 | 579 |  | infected contacts per case summed over layers, cases with no record = 0 (E80 upper definition); estimator used by verify_phase_d |
| 4.7 | R, upper definition | 0.4283 | 579 |  | infected contacts per case summed over layers, cases with no record = 0 (E80 upper definition) |
| 4.7 | share infecting >= 3, upper definition | 0.04318 | 579 |  | infected contacts per case summed over layers, cases with no record = 0 (E80 upper definition) |
| 4.7 | max infected, upper definition | 8 | 579 |  | infected contacts per case summed over layers, cases with no record = 0 (E80 upper definition) |
| 1.1 | offspring k, lower definition, MLE | 0.1079 | 579 |  | recorded infector -> infectee links per case, 0 otherwise (E80 lower definition) |
| 4.7 | R, lower definition | 0.04663 | 579 |  | recorded infector -> infectee links per case, 0 otherwise (E80 lower definition) |
| 4.7 | share infecting >= 3, lower definition | 0.001727 | 579 |  | recorded infector -> infectee links per case, 0 otherwise (E80 lower definition) |
| 4.7 | max infected, lower definition | 3 | 579 |  | recorded infector -> infectee links per case, 0 otherwise (E80 lower definition) |
| 1.18 | community median | 7.5 | 38 | 4–32 | municipality-layer total contacts per index case, rows with a known uninfected count and total > 0 (E77); interval = bootstrap 95% |
| 1.19 | community p90 | 195.1 | 38 | 68–822 | municipality-layer total contacts per index case, rows with a known uninfected count and total > 0 (E77); numpy linear percentile |
| 1.2 | community p90 / median | 26.01 | 38 | 5.565–93.13 | municipality-layer total contacts per index case, rows with a known uninfected count and total > 0 (E77); interval = bootstrap 95% (E77) |
| 1.21 | community zero share among rows with a known uninfected count | 0 | 38 |  | municipality-layer rows with total = 0 / rows with a known uninfected count |
| 3.12 | community tail power-law exponent (discrete) | 1.668 | 17 |  | Clauset 2009 discrete MLE above x_min, KS = 0.083 |
| 3.13 | community tail x_min | 21 | 38 |  | x_min minimising the KS distance (Clauset 2009) |
| 3.9 | coworker close contacts per index case, median | 2 | 8 |  | infected + uninfected coworkers per index case with any coworker record |

## 7. 社區接觸改為 scale-free（B62、B63，2026-10-16）

依教授 2026-10-12 指示（todolist M3），社區接觸數改為每位病例從單一分布 P(N = n) ∝ (n + s)^−α 抽一次，取消「是否參加大型活動」。

| # | 量 | 舊值 | 新值 | 來源 | 狀態 |
|---|---|---|---|---|---|
| 3.12' | 社區接觸冪律指數 α（P[204]，新參數） | —（舊模型為大型活動規模指數 1.67） | **1.74**（鎖定） | Fu 2010 community-only、≥15 分鐘、每人每日的最大概似估計（bootstrap 95% 1.62–1.88）；疫調尾巴 1.67 在區間內 | 📐 |
| 3.11、3.13–3.16 | 大型活動 P[199..203]、Gamma 離散度 P[198] | 校準／鎖定 | 鎖定不用（P[199] = 0） | B62 取消二元活動 | 🔧 |
| P[28] | 社區平移量 s | 平均接觸數 [1, 40] | 平移量 **[0.5, 20]** | 中位數約等於 s | 🔧 |
| P[29] | 社區接觸隔天再碰面機率下限 | 0.5（無來源） | **0** | Fu 2010：社區接觸只有 28.9% 天天碰面 | 📐 |
| 1.5 | daily_municipality | [1.20, 1.80] | **[1.20, 7.12]** | 上限為同一擬合分布去掉日誌 40 筆上限的平均（日誌上限時 2.08） | 📐 |
| 新 | community_day_zero_share | — | **[0.598, 0.641]** | Fu 2010 單日 0 次 62.0% 的 bootstrap 95% | 📐 |
| 新 | community_day_p90 | — | **[5, 8]** | Fu 2010 單日 P90 6（5–6），去掉上限為 8 | 📐 |
| 1.18、1.20 | community_median、community_tail_ratio | 計費 | **只量** | 疫調 38 筆為選擇樣本（E88） | — |

計算：covid19-research `scripts/summarize_taiwan_contact_data.py`（`by_setting`、`fit_shifted_power_law`）；可達性：`scripts/probes/probe_community_b62.py`（E100、B63）。

---

## 查證紀錄

| 日期 | 項目 | 結果 |
|---|---|---|
| 2026-10-12 | 3.8、4.13（Chen 2022） | 全文（PMC9296995）與補充資料查證：沒有依公司規模的群聚大小；「2.7 倍」是跨行業極值錯配 |
| 2026-10-12 | 3.12（冪律指數） | 38 筆資料以 Clauset 方法重算：k_min = 21、尾巴 17 筆、指數 1.67（離散）／1.68（連續） |
| 2026-10-12 | 1.2、1.4（每日接觸） | 2020 調查原文：每人最多記 12 位、問昨天、密切接觸定義為 2 公尺內交談或互動 15 分鐘以上 |
| 2026-10-12 | B5（city term） | Fu 2012 Results 一句話：北 12.4、南 12.3、東 9.5（全場所，無區域定義、無樣本數） |
| 2026-10-15 | 3.8、4.10、B4、B5 | 存入 13 篇論文（`lit/`），全部引文經程式比對確認存在於存檔；城市效應與職場的原文與小結見 [`source_quotes.md`](source_quotes.md) |
| 2026-10-15 | 全部 📐 項目 | 以 `recompute_taiwan_targets.py` 重算（§6）。重現：1.1、1.7、1.9、1.10、1.18–1.20、1.25–1.29、3.12–3.13、4.7。新發現：1.8 舊值分子分母母體不一致；3.9 樣本是 8 不是 6；4.8 可直接由資料算（12.9 天）；`verify_phase_d` 的 k 用動差法 |
| 2026-10-15 | 1.2–1.5、1.21 | Fu 2010 依情境與 ≥15 分鐘重算（§5）；職場、社區落在現行區間內，家戶、學校高於現行區間 |
| 2026-10-16 | 社區（B62、B63） | 社區接觸改為 scale-free，α = 1.74 來自 Fu 2010；社區計費目標改為 Fu 2010 單日 0 次比例與 P90；可達性通過（§7） |
