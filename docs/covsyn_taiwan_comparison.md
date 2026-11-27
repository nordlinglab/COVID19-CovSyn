---
title: "CovSyn production 主 run 與台灣實際數據的比較"
author: "Rex Lee"
date: 2026-11-27
version: "1.0"
license: "CC-BY-4.0"
purpose: "列出 Task 10 主 run 的每一項 checklist 結果，與台灣實際數據或文獻值、目標範圍及其出處並列。"
lang: zh-tw
---

# CovSyn production 主 run 與台灣實際數據的比較

## 說明

- 資料：Task 10 主 run `20261126-093307_dfac05e_production-gplan-n24-a0p4-r1` 的 checklist（`verify_phase_d`），用 checklist 選出的第 7 號候選模擬台灣第一波情境 1,000 次，共 1,779 個病例。
- 「CovSyn」欄是模擬的點估計，括號內是 bootstrap 95% 區間。
- 「實際值」是台灣資料或文獻報告的數字；只有目標範圍、沒有單一實際值的項目寫「—」。
- 判定：ok 表示點估計在目標範圍內；ok~ 表示點估計在範圍外，但 95% 區間與範圍重疊；FAIL 表示兩者都不重疊；「參考」表示只列出、不判定。
- 目標範圍與出處取自 checklist 的註記、[`calibration_sources.md`](calibration_sources.md)、決策紀錄 [`covsyn_decisions.md`](covsyn_decisions.md) 與程式註解。
  「範圍未記錄依據」表示實際值有出處，但範圍的兩端是 Phase D 訂的，沒有記下怎麼由實際值訂出。
- 主 run 判定 54 項：52 項通過（其中 4 項 ok~），2 項 FAIL；另有 51 項參考。
  以另一組獨立種子重跑（`checks_independent.txt`）結果相同：52 項通過（6 項 ok~），FAIL 的是同樣兩項。

## 1. 族群結構

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 病例的其他同住家人數 | 2.781（2.64–2.91） | 2.66 | 2.6–3 | 內政部戶政司（n.d.）110 年底縣市戶數結構（`data/demographic_data/縣市戶數結構.xls`），以人口加權；6 人以上家戶取平均 7 人時，總人口 2,337 萬與 2021 年底人口相符；範圍未記錄依據 | ok |
| 獨居比例 | 13.9%（11.7–16.1） | 13.4% | 11–16% | 內政部戶政司（n.d.），同上，以人口加權；範圍未記錄依據 | ok |
| 班級人數，國小 | 24.778（23.5–26.1） | 25.1 | 21–27 | 教育部統計處（n.d.）110 學年各級學校基本資料（`data/demographic_data/國民小學校別資料.xls`），以學生人數加權；範圍未記錄依據 | ok |
| 班級人數，國中 | 31.250（28.5–34.1） | 28.3 | 24–31 | 教育部統計處（n.d.）（`國民中學校別資料.xlsx`），以學生人數加權；範圍未記錄依據 | ok~ |
| 班級人數，高中 | 34.182（31.6–37.1） | 33.6 | 29–36 | 教育部統計處（n.d.）（`高級中等學校校別資料檔.xls`），以學生人數加權；範圍未記錄依據 | ok |
| 班級人數，大學 | 79.413（65.7–94.2） | 89.2 | 75–100 | 教育部統計處（n.d.）（`大專校院各校科系別學生數.xlsx`，學士班各系各年級），以學生人數加權；範圍未記錄依據 | ok |
| 台灣情境病例平均年齡 | 38.334（38.3–38.4） | 37.1；37.6（36.3–38.9） | 33–42 | 台灣個案資料 577 例（Wu & Nordling, 2024）：37.1 是年齡組內依人口分配單歲，37.6 是取組中點重算；範圍未記錄依據 | ok |

## 2. 接觸數

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 發病前每日同事接觸，有工作者 | 4.675（4.05–5.26） | 4.96（n = 416） | 4.34–5.61 | Fu 2010 接觸日記（Fu et al., 2012, 2022），工作日、≥ 15 分鐘、每天見面，以 2010 年人口加權 | ok |
| 發病前每日接觸，家庭 | 2.227（2.11–2.34） | 2.27；3.28 | 2.27–3.28 | 下限：2020 疫情期調查縱貫研究基線期的家庭接觸 2.27 ± 0.03（Wu et al., 2025, Table 4）；上限：Fu 2010 ≥ 15 分鐘（Fu et al., 2012, 2022） | ok~ |
| 發病前每日接觸，學校 | 1.280（1.06–1.55） | 1.72 | 1.15–1.72 | 上限：Fu 2010 ≥ 15 分鐘（Fu et al., 2012, 2022）；下限：2020 調查（Wu et al., 2025）依 Fu et al.（2012）的場所比例拆出的學校接觸（B61） | ok |
| 發病前每日接觸，社區 | 2.442（1.94–3.14） | 1.76 | 1.2–7.12 | 下限：舊目標，2020 調查（Wu et al., 2025）依 Fu et al.（2012）場所比例拆分；上限：Fu 2010 ≥ 15 分鐘社區接觸的冪律分布（α 1.737），由日記上限 40 外推到 CovSyn 上限 1,000（B63、E100）；1.76 是日記上限下的平均，為下界 | ok |
| 每日接觸，所有層 | 7.162（6.6–7.82） | 6.03；8.40 | 6.03–8.4 | 2020 調查縱貫研究基線期的總接觸 6.03 ± 0.06（Wu et al., 2025, Table 4）；Fu 2010 日記 ≥ 15 分鐘 8.40（Fu et al., 2012, 2022） | ok |
| 每天見到每位家人的比例 | 82.762%（82.1–83.5） | — | 50–100% | Fu et al.（2012） | ok |
| 社區單日接觸為 0 的比例 | 61.171%（58.1–63.6） | 62.0%（n = 1,954） | 59.8–64.1% | Fu 2010 ≥ 15 分鐘、所有日（Fu et al., 2012, 2022），bootstrap 95% | ok |
| 社區單日接觸 p90 | 5（4–6） | 6 | 5–8 | Fu 2010 ≥ 15 分鐘（Fu et al., 2012, 2022） | ok |
| 工作日同事接觸為 0 的比例 | 24.750%（20.4–29） | 27.4%（n = 416） | 23.2–31.8% | Fu 2010（Fu et al., 2012, 2022） | ok |
| 工作日同事接觸 p90 | 14（11–16） | 13 | 10–16 | Fu 2010（Fu et al., 2012, 2022） | ok |
| 城市對社區接觸的影響，置換檢定 p 值 | 0.890 | 沒有城市效應 | ≥ 0.05 | 設計要求（E4、E83）；ISSP 2017 與 Fu 2010 都不支持城市效應 | ok |
| 社區單日接觸，平均 | 2.483（2.03–3.05） | 1.76 | — | Fu 2010（Fu et al., 2012, 2022） | 參考 |
| 工作日同事接觸，平均 | 4.779（4.07–5.41） | 4.96 | — | Fu 2010（Fu et al., 2012, 2022） | 參考 |
| 工作群組人數，中位數 | 8（6.49–8） | — | — | 模型內部量 | 參考 |

## 3. 接觸紀錄的完整性

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 有候選接觸卻沒有有效接觸紀錄的病例 | 0（0–0） | — | 0 | 模型內部檢查（E5） | ok |
| 任何層都沒有接觸的病例 | 0.281%（0.105–0.547） | — | 0–4% | 模型內部檢查 | ok |

## 4. 傳播

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 每個接觸的累計二次感染率，學校 | 1.957%（1.35–2.82） | 3/126 = 2.4% | 1–4% | 錨點：Huang et al.（2021）Table 1 兩個教室群聚 1/69 與 2/57（E11，程式用 2.3%）；範圍 1–4% 沿用原版 CovSyn 的界限，原版當每日機率用，這裡改為累計（B4） | ok |
| 每個接觸的累計二次感染率，職場 | 3.432%（2.65–4.44） | 3.4% | 1.5–5% | 中心：Chen et al.（2022）英格蘭所有企業群聚攻擊率中位數（E11）；上限由原版的 15% 降為 5%，下限 1.5% 未記錄依據（`sar_anchors.py`） | ok |
| 每個 index case 的感染數，家庭 | 0.258（0.214–0.309） | 0.10 | 0.135–0.519 | Cheng et al.（2020） | ok |
| 每個 index case 的感染數，醫療照護 | 0.031（0.019–0.0421） | 0.06 | 0.062–0.368 | Cheng et al.（2020） | **FAIL** |
| 每個 index case 的感染數，社區 | 0.081（0.049–0.114） | 0.01 | 0.001–0.085 | Cheng et al.（2020）；上限取中心的 3 倍（E57） | ok |
| 每個境外移入種子的感染數 | 0.049（0.0481–0.0499） | 4/71 = 0.056 | 0.015–0.14 | Cheng et al.（2020），exact Poisson 95% | ok |
| 散布係數 k | 0.263（0.216–0.33） | 0.286（n = 579） | 0.2–0.35 | 台灣個案資料（Wu & Nordling, 2024），負二項 MLE | ok |
| R（每個 index case 的平均感染數） | 0.549（0.477–0.635） | 18/29 = 0.62 | 0.37–0.98 | Cheng et al.（2020）本土 index case；範圍取台灣兩種連結定義（E80） | ok |
| 感染 3 人以上的病例比例 | 6.700%（5.3–8.2） | 1–7 例／55 | 1.8–12.8% | 台灣 55 例本土病例（Wu & Nordling, 2024），記錄連結與確認接觸兩種定義（B83） | ok |
| 579 例中最大感染 8 人的百分位 | 3.601% | 最大 8 人 | 2.5–97.5 | 台灣個案資料（Wu & Nordling, 2024，確認接觸，上界，E80） | ok |
| 每個接觸的累計二次感染率，家庭 | 9.437%（8.04–11.1） | 錨點 10.88% | — | Cheng et al.（2020）每例感染數 ÷ CovSyn 候選接觸數（E55） | 參考 |
| 每個接觸的累計二次感染率，醫療照護 | 0.595%（0.373–0.812） | 錨點 2.92% | — | Cheng et al.（2020）÷ CovSyn 候選接觸數 | 參考 |
| 每個接觸的累計二次感染率，社區 | 0.417%（0.278–0.525） | 錨點 0.16% | — | Cheng et al.（2020）÷ CovSyn 候選接觸數 | 參考 |
| 在醫療照護中感染 1 人以上的 index case | 2.800%（1.8–3.8） | 5/55 = 9.1%（3.0–20.0） | — | 台灣第一波本土病例（Wu & Nordling, 2024） | 參考 |
| 在醫療照護中感染 2 人以上的 index case | 0.300%（0–0.7） | 1/55 = 1.8%（0.0–9.7） | — | 台灣第一波本土病例（Wu & Nordling, 2024） | 參考 |
| 第一波本土病例數，每次模擬 | 46.742（45.9–47.7） | 55 | — | 台灣第一波（Wu & Nordling, 2024） | 參考 |
| 第一波死亡數，每次模擬 | 2.606（2.52–2.69） | 3 | — | 台灣第一波（Wu & Nordling, 2024） | 參考 |

## 5. 社區接觸的分布

社區接觸的實際值來自台灣疫調的 38 筆 friend 與 other 接觸紀錄，是選擇性樣本（E88）。
這些紀錄由 `extract_tracing_reference.py` 從 forecasting repo 的活頁簿擷取，沒有公開出處。

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 每個病例的社區接觸數，中位數 | 2（2–3） | 7.5（n = 38） | — | 台灣疫調接觸紀錄 | 參考 |
| 每個病例的社區接觸數，p90 | 51（44–64） | 195（n = 38） | — | 台灣疫調接觸紀錄 | 參考 |
| 社區接觸數 p90 ÷ 中位數 | 10.200（9.4–15.6） | 26.0（5.5–93.1） | — | 台灣疫調接觸紀錄 | 參考 |
| 每個病例的社區接觸數，最大值 | 889（765–889） | 850 | — | 台灣疫調接觸紀錄 | 參考 |

## 6. 疫調時窗內的接觸數

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 每個 index case 在 Jian 疫調時窗內的接觸數 | 24.204（21.2–27.6） | 1,435/55 = 26.1 | — | Jian et al.（2020），本土病例 | 參考 |
| 每個確診病例的接觸數，所有病例 | 14.529（14.4–14.7） | 16.5（13.9–19.1） | — | Jian et al.（2020），487 例 | 參考 |
| 每個本土病例的接觸數，每次模擬 | 19.330（18.8–19.7） | 26.1 | — | Jian et al.（2020） | 參考 |
| 每個境外移入病例的接觸數，每次模擬 | 13.764（13.6–13.9） | 6,616/432 = 15.3 | — | Jian et al.（2020） | 參考 |
| 每個 index case 的醫療照護接觸數，Jian 時窗內 | 2.010（1.85–2.2） | 9.89 | — | Jian et al.（2020），醫院或診所接觸 | 參考 |
| 發病前就開始的接觸 | 57.206%（53.2–60.1） | 27.5% | — | Cheng et al.（2020）；判定為紀錄方式造成（E84） | 參考 |

## 7. 病程時間（天）

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 潛伏期平均，文獻 | 5.736（5.59–5.9） | 3.9–8.0 | 3.9–8 | 17 篇研究報告的平均值範圍（`validation_reference/literature_provenance.csv`，取自 Cheng et al., 2020; Park et al., 2020; Quesada et al., 2021）：下限 3.9 是 Ki（2020）原文；上限 8.0 是 Quesada et al.（2021）Table 1 中 Xu 等人 2020 年 3 月 7 日的研究 | ok |
| 潛伏期平均，台灣 | 5.736（5.59–5.9） | 6.2（n = 28） | 4.96–7.57 | 台灣個案資料（Wu & Nordling, 2024），bootstrap 95% | ok |
| 發病前可傳染期，平均 | 1.449（1.39–1.5） | 0.5–4 | 1–3 | Byrne et al.（2020）發病前傳染期（T3）表中 13 筆估計，Huang 等人的 4 天以外都在 0.5–2.9 天（E46） | ok |
| 發病前可傳染期為 0 天的比例 | 9.694%（7.75–11.7） | — | 0–12% | 無文獻出處：Phase D 的設計值，用來消除舊模型 20% 病例窗口為 0 的現象（B22、E96） | ok |
| 潛伏感染期（latent）平均 | 4.203（4.09–4.32） | 4.1；5.5 | 4.1–5.5 | 4.1：Cheng et al.（2020）的中位數；5.5：Xin et al.（2022）的平均數 | ok |
| 發病到確診，中位數 | 6（6–6） | 6（n = 442） | 5–7 | 台灣個案資料（Wu & Nordling, 2024）；Ge et al.（2021）發病到隔離 5 天 | ok |
| 感染到隔離，平均 | 11.198（11–11.4） | — | 8–14 | 推算：潛伏期加上台灣發病到確診中位數 5–7 天 | ok |
| 感染到確診，平均，台灣 | 11.215（11–11.4） | 12.9（n = 33） | 11.4–14.6 | 台灣個案資料（Wu & Nordling, 2024），bootstrap 95% | ok~ |
| 確診到結案，有症狀 | 26.176（25.3–27.1） | 27.0（n = 56） | 22–32 | 台灣個案資料（Wu & Nordling, 2024，E91） | ok |
| 確診到結案，無症狀 | 19.296（17.7–21） | 23.3（n = 23） | 17–31 | 台灣個案資料（Wu & Nordling, 2024，E91） | ok |
| ICU 到結案 | 31.354（28.5–34.1） | 36.2（n = 28） | 29–44 | 台灣個案資料（Wu & Nordling, 2024，E93） | ok |
| 發病到進 ICU | 9.358（7.8–11） | 7.2（n = 40） | 5.4–9.2 | 台灣個案資料（Wu & Nordling, 2024，E92） | ok~ |

## 8. 醫療照護接觸的時間點

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 發病第 8 天以後開始的醫療照護接觸 | 24.353%（23.4–25.6） | 256/697 = 36.7% | 20–50% | Cheng et al.（2020） | ok |
| 第 4 天以前開始的醫療照護接觸 | 8.549%（7.7–9.49） | 33.9% + 21.5% = 55.4% | 40–70% | Cheng et al.（2020），發病前加發病 0–3 天 | **FAIL** |
| 每個 index case 的醫療照護接觸數 | 5.206（4.98–5.45） | — | ≥ 3.335 | 下限：Cheng et al.（2020）醫療感染數下限 ÷ 每接觸感染率上限 | ok |

## 9. 嚴重度

| 項目 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 無症狀比例 | 21.600%（19.4–24） | 118/578 = 20.4% | 20–28% | 台灣個案資料（Wu & Nordling, 2024，B61） | ok |
| 有症狀者進 ICU | 12.117%（9.89–14.6） | 58/460 = 12.6% | 9–17% | 台灣個案資料（Wu & Nordling, 2024，B61） | ok |
| ICU 病人死亡 | 13.684%（7.01–19.6） | 7/58 = 12.1% | 7–19% | 台灣個案資料（Wu & Nordling, 2024） | ok |
| 病例致死率 | 1.300%（0.6–2） | 7/578 = 1.21% | 0.7–2% | 台灣個案資料（Wu & Nordling, 2024） | ok |
| ICU 率比，60+ ÷ 20–39 歲 | 5.217（3.2–10.8） | — | 3–12 | Verity et al.（2020）的年齡梯度 | ok |
| 有症狀者進 ICU，0–19 歲 | 0%（0–0） | — | — | n = 172 | 參考 |
| 有症狀者進 ICU，20–39 歲 | 5.607%（2.73–8.49） | — | — | n = 269 | 參考 |
| 有症狀者進 ICU，40–59 歲 | 11.475%（7.93–15.5） | — | — | n = 311 | 參考 |
| 有症狀者進 ICU，60+ 歲 | 29.255%（22.2–35.7） | — | — | n = 248 | 參考 |

## 10. 年齡別感染風險比（20–39 歲 = 1）

在家庭、醫療照護、社區三層量測（Cheng 2020 追蹤的場所），20,000 個 index case、7,467 個感染。

| 年齡 | CovSyn（95% 區間） | 實際值 | 目標範圍 | 出處 | 判定 |
|---|---|---|---|---|---|
| 0–19 | 0.419（0.363–0.484） | 0.34；0.56；0.58；0.59 | 0.34–0.77 | Zhang et al.（2020）0.34；Viner et al.（2021）未滿 20 歲成為受感染接觸者的 OR 0.56（0.37–0.85）；Uthman et al.（2024）原始株兒童與青少年 OR 0.58（0.44–0.77，19 篇）；Madewell et al.（2020）家戶 SAR 兒童 16.8% ÷ 成人 28.3% ≈ 0.59；上限 0.77 是 Uthman 區間上限 | ok |
| 20–39 | 1.000（0.908–1.1） | 1 | 1 | 基準組 | ok |
| 40–59 | 1.755（1.62–1.9） | 1.83 ± 0.10 | 1.63–2.03 | Cheng et al.（2020） | ok |
| 60+ | 1.329（1.21–1.46） | 1.32 ± 0.10 | 1.12–1.52 | Cheng et al.（2020） | ok |

## 11. 與台灣不符或未驗證的地方

1. 兩項 FAIL 都在醫療照護層：每個 index case 的醫療照護感染數只有 Cheng et al.（2020）的一半，醫療照護接觸開始得也太晚。
2. 規模偏小：第一波本土病例 46.7 例，實際 55 例；疫調時窗內的接觸數比 Jian et al.（2020）少約 7–26%。
3. 獨立種子重跑時，「ICU 率比 60+ ÷ 20–39 歲」為 33.232，遠超出 3–12，卻仍判為 ok~；這個指標的區間太寬，判定沒有鑑別力。
4. 沒有文獻出處的目標只有一項：發病前可傳染期為 0 天的比例 0–12%，是 Phase D 的設計值（B22、E96）。
5. 下列範圍的實際值有出處，但範圍兩端是 Phase D 訂的，沒有記下怎麼訂：家戶其他成員 2.6–3、獨居 11–16%、各學制班級人數、病例平均年齡 33–42、職場二次感染率下限 1.5%。
6. 潛伏期清單與 Park et al.（2020）Table 3 有三處不同：Ki 為 3.9（原文；Park 列 22 例的 3.6）、Backer 為 6.5（Park 列 6.4）、Lauer 為 5.1（Park 列 5.2）。
   範圍的兩端 3.9 與 8.0 都已核對到原文或 Quesada et al.（2021），不受影響。
7. 0–19 歲風險比的參考值有兩點要在論文中說明：Viner et al.（2021）與 Uthman et al.（2024）報告的是勝算比（OR），不是風險比；Uthman et al.（2024）只算 0–19 歲年齡組的研究時，OR 為 0.43（0.24–0.75，10 篇）。
8. 社區接觸分布的 38 筆疫調紀錄取自 Wu 與 Nordling 資料集的完整活頁簿，沒有公開版本。

## References

Byrne, A. W., McEvoy, D., Collins, A. B., Hunt, K., Casey, M., Barber, A., Butler, F., Griffin, J., Lane, E. A., McAloon, C., O'Brien, K., Wall, P., Walsh, K. A., & More, S. J. (2020). Inferred duration of infectious period of SARS-CoV-2: Rapid scoping review and analysis of available evidence for asymptomatic and symptomatic COVID-19 cases. *BMJ Open, 10*(8), Article e039856. https://doi.org/10.1136/bmjopen-2020-039856

Chen, Y., Aldridge, T., Ferraro, C., & Khaw, F.-M. (2022). COVID-19 outbreak rates and infection attack rates associated with the workplace: A descriptive epidemiological study. *BMJ Open, 12*(7), Article e055643. https://doi.org/10.1136/bmjopen-2021-055643

Cheng, H.-Y., Jian, S.-W., Liu, D.-P., Ng, T.-C., Huang, W.-T., Lin, H.-H., & Taiwan COVID-19 Outbreak Investigation Team. (2020). Contact tracing assessment of COVID-19 transmission dynamics in Taiwan and risk at different exposure periods before and after symptom onset. *JAMA Internal Medicine, 180*(9), 1156–1163. https://doi.org/10.1001/jamainternmed.2020.2020

Fu, Y.-C., Wang, D.-W., & Chuang, J.-H. (2012). Representative contact diaries for modeling the spread of infectious diseases in Taiwan. *PLoS ONE, 7*(10), Article e45113. https://doi.org/10.1371/journal.pone.0045113

Fu, Y.-C., Wang, D.-W., Chuang, J.-H., & Fu, H. (2022). *Social contact data for Taiwan* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6385759

Ge, Y., Martinez, L., Sun, S., Chen, Z., Zhang, F., Li, F., Sun, W., Chen, E., Pan, J., Li, C., Sun, J., Handel, A., Ling, F., & Shen, Y. (2021). COVID-19 transmission dynamics among close contacts of index patients with COVID-19: A population-based cohort study in Zhejiang Province, China. *JAMA Internal Medicine, 181*(10), 1343. https://doi.org/10.1001/jamainternmed.2021.4686

Huang, Y.-T., Tu, Y.-K., & Lai, P.-C. (2021). Estimation of the secondary attack rate of COVID-19 using proportional meta-analysis of nationwide contact tracing data in Taiwan. *Journal of Microbiology, Immunology and Infection, 54*(1), 89–92. https://doi.org/10.1016/j.jmii.2020.06.003

Jian, S.-W., Cheng, H.-Y., Huang, X.-T., & Liu, D.-P. (2020). Contact tracing with digital assistance in Taiwan's COVID-19 outbreak response. *International Journal of Infectious Diseases, 101*, 348–352. https://doi.org/10.1016/j.ijid.2020.09.1483

Ki, M. (2020). Epidemiologic characteristics of early cases with 2019 novel coronavirus (2019-nCoV) disease in Korea. *Epidemiology and Health, 42*, Article e2020007. https://doi.org/10.4178/epih.e2020007

Madewell, Z. J., Yang, Y., Longini, I. M., Halloran, M. E., & Dean, N. E. (2020). Household transmission of SARS-CoV-2: A systematic review and meta-analysis. *JAMA Network Open, 3*(12), Article e2031756. https://doi.org/10.1001/jamanetworkopen.2020.31756

Park, M., Cook, A. R., Lim, J. T., Sun, Y., & Dickens, B. L. (2020). A systematic review of COVID-19 epidemiology based on current evidence. *Journal of Clinical Medicine, 9*(4), Article 967. https://doi.org/10.3390/jcm9040967

Quesada, J. A., López-Pineda, A., Gil-Guillén, V. F., Arriero-Marín, J. M., Gutiérrez, F., & Carratala-Munuera, C. (2021). Incubation period of COVID-19: A systematic review and meta-analysis. *Revista Clínica Española (English Edition), 221*(2), 109–117. https://doi.org/10.1016/j.rceng.2020.08.002

Uthman, O. A., Lyngse, F. P., Anjorin, S., Hauer, B., Hakki, S., Martinez, D. A., Ge, Y., Jonnerby, J., Julin, C. H., Lin, G., Lalvani, A., Loss, J., Madon, K. J., Martinez, L., Næss, L. M., Page, K. R., Prieto, D., Robertson, A. H., Shen, Y., … Buchholz, U. (2024). Susceptibility and infectiousness of SARS-CoV-2 in children versus adults, by variant (wild-type, alpha, delta): A systematic review and meta-analysis of household contact studies. *PLoS ONE, 19*(9), Article e0306740. https://doi.org/10.1371/journal.pone.0306740

Verity, R., Okell, L. C., Dorigatti, I., Winskill, P., Whittaker, C., Imai, N., Cuomo-Dannenburg, G., Thompson, H., Walker, P. G. T., Fu, H., Dighe, A., Griffin, J. T., Baguelin, M., Bhatia, S., Boonyasiri, A., Cori, A., Cucunubá, Z., FitzJohn, R., Gaythorpe, K., … Ferguson, N. M. (2020). Estimates of the severity of coronavirus disease 2019: A model-based analysis. *The Lancet Infectious Diseases, 20*(6), 669–677. https://doi.org/10.1016/S1473-3099(20)30243-7

Viner, R. M., Mytton, O. T., Bonell, C., Melendez-Torres, G. J., Ward, J., Hudson, L., Waddington, C., Thomas, J., Russell, S., van der Klis, F., Koirala, A., Ladhani, S., Panovska-Griffiths, J., Davies, N. G., Booy, R., & Eggo, R. M. (2021). Susceptibility to SARS-CoV-2 infection among children and adolescents compared with adults: A systematic review and meta-analysis. *JAMA Pediatrics, 175*(2), 143–156. https://doi.org/10.1001/jamapediatrics.2020.4573

Wu, M.-H., Fu, H., Wu, C.-Y., Chen, P.-C., Lin, H.-H., Chang, S.-S., & Chang, H.-H. (2025). Health beliefs and their relation to household, necessary, and unnecessary contacts across COVID-19 epidemic waves in Taiwan. *BMC Public Health, 25*(1), Article 4117. https://doi.org/10.1186/s12889-025-25340-1

Wu, Y.-H., & Nordling, T. E. M. (2024). A structured course of disease dataset with contact tracing information in Taiwan for COVID-19 modelling. *Scientific Data, 11*(1), Article 821. https://doi.org/10.1038/s41597-024-03627-z

Xin, H., Li, Y., Wu, P., Li, Z., Lau, E. H. Y., Qin, Y., Wang, L., Cowling, B. J., Tsang, T. K., & Li, Z. (2022). Estimating the latent period of coronavirus disease 2019 (COVID-19). *Clinical Infectious Diseases, 74*(9), 1678–1681. https://doi.org/10.1093/cid/ciab746

Zhang, J., Litvinova, M., Liang, Y., Wang, Y., Wang, W., Zhao, S., Wu, Q., Merler, S., Viboud, C., Vespignani, A., Ajelli, M., & Yu, H. (2020). Changes in contact patterns shape the dynamics of the COVID-19 outbreak in China. *Science, 368*(6498), 1481–1486. https://doi.org/10.1126/science.abb8001

內政部戶政司 [Department of Household Registration, Ministry of the Interior]. (n.d.). *人口統計資料* [Population statistics; data set]. Retrieved November 27, 2026, from https://www.ris.gov.tw/app/portal/346

教育部統計處 [Department of Statistics, Ministry of Education]. (n.d.). *各級學校基本資料* [Basic data of schools at all levels; data set]. Retrieved November 27, 2026, from https://depart.moe.edu.tw/ED4500/News_Content.aspx?n=5A930C32CC6C3818&sms=91B3AAE8C6388B96&s=8AF80DB14ADF7370
