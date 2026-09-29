# CovSyn 驗證：定義清單與各層結論（草稿）

> 對應 todo.md 的「Definition checklist」與 §7。資料：run5 參數，`synthetic_data_results_spread_Taiwan_weight_MC1000`（1,313 例）與 `synthetic_data_results_cheng2020_MC1000`（100,000 個 index case）。
> 圖與 CSV：`diagnostic_figures/validation_run5/`。決策與發現編號（B、C、E、F）對應 `covsyn_decisions.md`。
> **狀態：初判。** 「是否可接受」「是否需要改模型」都是建議，依 A1 要在 Phase C 由使用者確認後才動模型。
> 最後更新：2026-09-14
>
> ⚠️ **2026-09-21：以下內容描述的是 Phase D 改動前的模型**（run5 參數）。Phase D 已依 B14、B17–B36
> 改寫接觸與病程機制並重新訓練中，下列每一項的數字、定義與結論都要在重跑完成後逐條更新。
> 定義清單中至少這幾項已經改變：**context size**（社區層不再有情境大小，改為直接抽
> `Poisson(ν·λ)`，與縣市人口脫鉤，B27；職場改為「企業規模 → 工作群組」兩段式，B31）、
> **candidate contact**（期間不再一律到隔離日，醫療層延長到隔離後最多 14 天，B23；隔離日改由
> 疫調或自身症狀先到者決定，B26）、**layer-specific transmission probability**（移除每接觸者的
> overdispersion 樂透，改為每病例一個乘數 ν，B17）、**重症／死亡機率**（改為隨年齡變動，B20／B33）。
> 更新後的驗收數字見 `phaseD_checks.txt` 與 `verify_phaseD.py`。

---

## 一、定義清單（依程式碼實際行為）

| 名詞 | CovSyn 的定義 | 程式位置 |
|---|---|---|
| **Context size**（社交情境大小，① ） | 感染者所在情境的人數上限：household = 家戶人數 − 1；school = 班級人數（含自己）；workplace = work group size（工商普查企業規模）；health care = 診所每日就診人次；municipality = 所住縣市人口。todo 所說的「upper bound / opportunity set」對應的是這個量。 | `Draw_social_data`；`Data_synthesize.py:752` 的 `n` |
| **Candidate contact**（② ） | 從情境中以 `Binomial(context size, p0)` 抽出的人；每人**至少被指定一個接觸日**（找不到接觸日時依每日機率強制指定）。期間是感染日（day 0）到監測隔離日（`monitor_isolation_period`，含當天），**沒有時間長度門檻**。驗證圖中 candidate = `len({layer}_effective_contacts)`（C2）。 | `Data_synthesize.py:752`、`generate_first_contact_matrix`（618–668） |
| **Realized contact**（接觸日） | candidate × 日的接觸矩陣中值為 1 的格子，也就是「某人某天有接觸」。第一個接觸日之後，前一天有接觸的人當天接觸機率為 `p1`（連續接觸），否則為當天的 `daily_p`。 | `draw_social_contacts_each_day`（731–796） |
| **Contact probability** | 每層 7 個參數：`p0` 情境中的人成為 candidate 的機率；`p1` 連續接觸機率；`p2` 健康／無症狀時的每日接觸機率；`p3` 有症狀時的每日接觸機率；再加 steepness、symptom phase、width，以 logistic 曲線在發病前後從 `p2` 過渡到 `p3` 再回來。無症狀者整段用 `p2`。 | `generate_logistic_contact_p`（522–531） |
| **Layer-specific transmission probability**（每接觸日） | 某 candidate 在接觸日 *t* 被感染的機率 = `a_L(t) × RR_age / Z_L`（上限 1）；若該接觸抽到 overdispersion 狀態（機率 `rO`），再乘 `wO`（上限 1）。`a_L(t)` 是該層 25 個「daily secondary attack rate」參數，依病程映射到 latent 結束到 infectious 結束的每一天（latent 期間為 0，隔離日之後不存在接觸）。已有自然免疫或疫苗保護者不會被感染。取**第一個**抽中的接觸日為感染日。 | `calculate_daily_secondary_attack_rate`（867–905）、`draw_infection_status`（935–973）、`calculate_age_adjusted_secondary_attack_rate`（975–981） |
| **Effective contact** / **Secondary infection**（③ ） | 被這個感染者感染的 candidate，在 `{layer}_effective_contacts` 記為 1，感染日記在 `{layer}_effective_contacts_infection_time`。每個 effective contact 都會成為新的感染者並排入佇列。 | `draw_contact_data`（983 起）、`Data_synthesis_main.py:347–414` |
| **Infection event** | `transmission_digraph` 的一列：（感染源 ID、被感染者 ID、感染日、接觸層）。只有 `infection_day <= time_limit` 的事件會被模擬並寫入（spread_Taiwan_weight = 84 天，cheng2020 = 0 天即只模擬 index case）。 | `Data_synthesis_main.py:284–289` |
| **Attack rate（模型參數）** | 上面的 `a_L(t)`：**每個接觸日**的感染機率，不是累積 SAR。 | `input_P[70:195]` |
| **Calibration anchor（累積 SAR）** | 給優化器的每層「每個接觸者的累積 SAR」（lb、中心、ub），以 `p_daily = 1 − (1 − SAR)^(1/n_days)` 換算成每日值（B8）。 | `parameters_for_initialization.py:840` |
| **Secondary attack rate（驗證輸出）** | 某層 effective contacts 總數 ÷ 該層 candidate contacts 總數。與 Cheng 2020 比較時用 cheng2020 模式（只有一代、100 個 index case／次）。 | `validate_infection.py` |
| **Secondary attack rate（Cheng 2020）** | clinical：有症狀的次級病例 ÷ 密切接觸者；密切接觸者 = 未穿 PPE、面對面 > 15 分鐘；追蹤期間從發病日（必要時往前至多 4 天）到確診日。all infections 另加 4 名無症狀者（household 3、非同住親屬 1）。 | Cheng et al. 2020, Table 2 |
| **Endpoint inclusion rule** | 目前實作：只模擬感染日 ≤ `time_limit` 的感染者；超過的次級感染**仍記在感染源的 effective contacts 裡**，但不再模擬其病程與接觸。另外，若任一天確診數 ≥ 1,000 就停止，並刪除陽性日晚於該日的個案。**正式的 endpoint 規則依 C8 延後**到方法比較階段。 | `Data_synthesis_main.py:286`、`425–454` |

**用語提醒**：todo 說 candidate contacts 是 upper bound / opportunity set，這在 CovSyn 對應的是 ①（context size）。目前驗證圖裡的「candidate」是 ②（C2 已決定）；論文 Methods 要把 ①②③ 三個量分開寫清楚。

---

## 二、各層結論（todo §7 格式）

### Household — 初判：**接觸人數不合理，攻擊率合理**

1. **CovSyn**：context size 平均 1.63，candidate 1.50（index 模式 1.60），沒有家戶接觸的比例 34–39%；SAR 6.51%（index 模式）、6.11%（spread）；佔所有次級感染 38%。
2. **參考資料**：內政部 2021 戶數結構，以人口加權（隨機一個人所在的家戶）平均 2.78 名其他成員，獨居 13.5%；Cheng 2020 household SAR clinical 4.6%、所有感染 6.62%（10/151）；次級感染中 household（含非同住親屬）Cheng 68%（15/22），疫調本土被感染者 47%（9/19）。
3. **差異**：接觸人數少 41%，獨居比例高約 2.6 倍；SAR 幾乎一致；感染佔比偏低。
4. **可接受嗎**：接觸人數不行，SAR 可以。
5. **理由**：CovSyn 的 context size 幾乎等於「隨機一戶」（1.61），偏差來自依**戶數**抽樣（E1），是明確的機制錯誤，不是參考資料的不確定性。SAR 是校準目標，貼近是預期結果（循環，不能當獨立驗證）。
6. **要改模型嗎**：建議要（E1：改成人口加權抽家戶大小）。改了之後家戶接觸變多，要重新校準 household 攻擊率。

### School — 初判：**情境大小不合理，接觸人數不確定**

1. **CovSyn**：每位學生的同班人數 國小 16.3、國中 23.8、高中 23.8、大學 58.9；candidate 國小 2.8、國中 3.3、高中 3.6、大學 8.1（全部學生平均 4.8，中位數 4）；SAR 1.74%；佔次級感染 4%。
2. **參考資料**：教育部校別資料，學生加權同班人數 24.1、27.3、32.6、88.3；疫調「同校」接觸者每例中位數 5、平均 26（n=24，多為境外移入的同校團體）；2020 第一波期間學校 2/25 正常開學、沒有全面停課（1 例停班、2 例停校）；台灣疫調本土被感染者中 school 為 0/19。
3. **差異**：同班人數少 13–33%；candidate 只有班級的 12–14%，但與疫調中位數同數量級。
4. **可接受嗎**：情境大小不行；candidate 無法判斷。
5. **理由**：情境大小的偏差來自依**學校**抽樣而非學生（E2），與 household 同一種錯誤。「班上誰算密切接觸者」沒有台灣直接資料，疫調的同校接觸者多為旅遊團體、長尾明顯。期間對照沒有問題，但 84 天 spread 模擬沒有日曆日期，也沒有「確診後停課」機制。
6. **要改模型嗎**：建議修正抽樣權重（E2）；candidate 比例（`p0`）暫不動。

### Workplace — 初判：**不確定（接觸人數與小樣本疫調相符）**

1. **CovSyn**：work group size 平均 7.49；有職場接觸者的 candidate 平均 3.19、中位數 2；SAR 2.08%（低於錨點 3.4%）；佔次級感染 6%。
2. **參考資料**：疫調同事接觸者 第一波 平均 3.25、中位數 2（n=8），延伸到 2021 平均 11.7、中位數 3（n=22）；工商普查企業規模 以企業計 9.98、以員工計 139.4（只當抽樣診斷，C6）；Huang 2021 兩個職場群聚 3/41、2/24（挑選發布的群聚）；台灣本土被感染者 workplace 0/19。
3. **差異**：candidate 中位數一致；錨點 3.4% 出自英國研究（E11），不是台灣。
4. **可接受嗎**：暫時可以，標為 exploratory。
5. **理由**：沒有台灣 workgroup 資料（已全面搜尋）。抽樣規則與「以企業計」一致（E3），但企業規模不是工作群組，改成員工加權反而會讓接觸暴增、與疫調更不符，所以 E3 的修正方向本身不確定。
6. **要改模型嗎**：目前不建議；找到 workgroup 資料再決定。錨點出處要在論文更正（E11）。

### Health care — 初判：**感染佔比不合理，接觸人數與 SAR 可接受**

1. **CovSyn**：candidate 平均 6.51（有接觸者 8.14）；candidate 中醫療佔 29%；SAR 1.47%；佔次級感染 **40%**（index 模式 36%）。
2. **參考資料**：Cheng 2020 醫療接觸者 698/2,761 = 25%，SAR 0.9%（0.4–1.9），佔次級感染 27%（6/22）；Huang 2021 醫院群聚 8/455 = 1.76%；疫調「同醫院」接觸者每例中位數 1.5、平均 4.25（n=8）；疫調本土被感染者 16%（3/19）。
3. **差異**：接觸組成接近 Cheng；SAR 在 Cheng CI 內但比點估計高 63%；感染佔比比兩個台灣來源都高。
4. **可接受嗎**：接觸人數與 SAR 可以，感染佔比不行。
5. **理由**：接觸人數是用 Cheng 校準的（循環，不是獨立驗證）。40% 的佔比來自「醫療接觸多（29%）× SAR 偏高（1.47% vs 0.9%）」；若 SAR 回到 Cheng 點估計，佔比會明顯下降。疫調的醫療接觸者樣本太小，只能輔助。
6. **要改模型嗎**：建議 Phase C 考慮把 health care 攻擊率往 Cheng 點估計收（例如收窄上界），並重看 E6。

### Municipality — 初判：**接觸人數不合理（人口放大），SAR 不確定**

1. **CovSyn**：candidate 平均 12.5，與縣市人口完全成正比（r = 1.00，新北市約 24.7 人、最小縣市約 1.3 人）；SAR 0.27%；school + workplace + municipality 合計 SAR 0.51%；佔次級感染 11%。
2. **參考資料**：疫調「friend + other」接觸者每例中位數 2、平均 52（n=64，長尾到 850）；Cheng 2020「others」SAR 0.1%（0.0–0.3），佔次級感染 5%（1/22）；疫調本土被感染者 37%（7/19，多為 other/unknown 關係）；沒有找到台灣 COVID 期間的 mobility／社區接觸數資料。
3. **差異**：人口放大約 19 倍，沒有行為依據；合計 SAR 高於 Cheng CI（已知，B10）；兩個台灣來源的感染佔比互相矛盾。
4. **可接受嗎**：接觸人數不行；SAR 與佔比無法判斷。
5. **理由**：`Binomial(縣市人口, p_M)` 讓大城市居民的社區接觸數機械性地放大（E4），這是模型設計造成的，不是加總錯誤。municipality 錨點 0.2% 沒有文獻出處（E11）。
6. **要改模型嗎**：建議要：改成不隨人口放大的社區接觸分布，但需要參考資料（F6 每日總接觸人數是候選）。

### 跨層問題（不屬於單一層）

- **E5 bug**：模擬中的第一個病例若死亡就完全不會傳染（約 8% 的 spread 模擬），是程式錯誤，建議 Phase C 直接修。
- **E7／E8 追蹤窗口**：CovSyn 從感染日算、無時間門檻；Cheng 從發病日算、> 15 分鐘，校準目標本身窗口不一致。
- **E11 文獻出處**：錨點 household 10.1%、health care 0.4% 出自 Ge 2021（中國浙江），workplace 3.4% 出自 Chen 2022（英國），school 2.4% 出自 Huang 2021（程式用 2.3%），municipality 0.2% 沒有出處；`parameters_for_initialization.py` 的註解寫錯，論文要照正確出處引用。
