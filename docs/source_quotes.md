---
title: Verbatim source quotes behind CovSyn calibration values
author: Rex Lee
date: 2026-10-15
purpose: Holds, for each calibration value or design choice, the exact sentence or table cell of the stored source that supports it, so that every value can be traced to a paper rather than to our own judgement.
---

# 校正數值的原文證據

每一則引文都存在 `lit/` 裡的檔案中，並經程式比對確認逐字出現在該檔（2026-10-15 以程式比對，全部找到）。
兩則 Schläpfer 引文在 PDF 文字層中缺少希臘字母 β 與減號，因此是在忽略這兩個符號後比對成功。

格式：**引文**（位置）— 檔案 — 支持什麼。
標示「**我的推算**」的內容不是論文原文，是我根據原文數字算出來的，必須另外查證或標示為推算。

---

## 1. City term：接觸數是否隨城市人口增加

### 1.1 台灣

**Fu 2012**（citekey `2026-10-07-4c8c5035`；`lit/metadata/2026-10-07-4c8c5035_fu2012.xml`、`lit/source/2026-10-07-4c8c5035_Fu2012_….pdf`）

> "Within Taiwan, most regional differences are not significant (for example, the average number of contacts is 12.4 in the North and 12.3 in the South), except for those living in the more remote East (only 9.5)."（Results, Contact Patterns in Everyday Life）

> "Respondents were well distributed over all geographic regions and across different urbanization levels (about 13.6% rural and 86.4% urban) that matched the distributions of the total population."（Results, Characteristics of Study Samples）

> "…recorded every person with whom they had contacted during the past 24 hours (up to a maximum of 40 persons…"（同上）

- 原文**沒有**：北、南、東的定義（哪些縣市）、各區樣本數、信賴區間、依都市化程度的接觸數。
- 支持：北部和南部幾乎相同；東部較低。**不支持**「接觸數與城市人口成正比」。

### 1.2 其他國家的面對面接觸調查

**Read 2014**（南中國，廣州周邊 40 個社區；citekey `read2014MixingPatternsProcRSocBBiolSci`，ids `2026-10-15-b4052ee0`；`lit/metadata/2026-10-15-b4052ee0_PMC4024290.xml`）

> "There was no significant difference in the number of contacts when comparing administratively designated rural and urban study locations"（Results）

> "Our results suggest that measurable social contacts occur in similar numbers regardless of urbanization if locations share a cultural context"（Discussion）

- 同一段也報告：人口密度最高的地區接觸**較少**（部分因為這些地區的人較老）。

**Béraud 2015**（法國全國調查；`2026-10-15-c249f637`；`lit/metadata/2026-10-15-c249f637_PMC4503306.xml`）

> "Region of residency did not influence the number of contacts."（Results）

**Kiti 2014**（肯亞；`2026-10-15-d5821dac`；`lit/metadata/2026-10-15-d5821dac_PMC4134222.xml`）

> "When stratified by residence, participants in the rural areas reported higher mean number of contacts (18.8/person/day, Part A of Figure 3 ) compared with their semiurban counterparts (16.5/person/day…"（Results）

- 鄉村**多於**半都市（方向與正比假設相反）。

**Prem 2021**（文獻回顧段落；`2026-10-15-a6764d90`；`lit/metadata/2026-10-15-a6764d90_PMC8354454.xml`）

> "In these studies, individuals in rural settings documented more contacts at home than their urban counterparts"（Introduction）

> "However, individuals in rural settings in Zimbabwe [ 30 ] reported a lower total number of contacts than those in peri-urban settings."（Introduction）

**Mousa 2021**（中低收入國家接觸調查的系統性回顧；`2026-10-15-6bbd5d22`；`lit/metadata/2026-10-15-6bbd5d22_PMC8765757.xml`）

> "…only three studies ( Kiti et al., 2014 ; le Polain de Waroux et al., 2018b ; Neal et al., 2020 ) contained information from both rural and urban sites, allowing classification."（Discussion）

- 系統性回顧本身說明：城鄉比較的資料很少，無法下結論。

### 1.3 隨城市人口縮放的定量證據（手機資料）

**Schläpfer 2014**（葡萄牙手機、英國市話；`2026-10-15-dd1a3474`；`lit/source/2026-10-15-dd1a3474_schlapfer2014_arxiv.pdf`，arXiv v3）

> "…with exponent β = 1.12 > 1 (95% confidence interval (CI) [1.11,1.14])."（Section II.A，p. 4）

> "…every doubling of a city's population results, on average, in approximately 12% more mobile phone contacts per person…"（Section II.A，p. 4）

> "Other city definitions and shorter observation periods [27] lead to similar results with overall β = 1.05 − 1.15 (95% CI [1.00,1.20])."（Section II.A，p. 5）

- 注意：這是 **15 個月累積的手機通話對象數**，不是每天面對面接觸。
- 論文的 β 是「全市總接觸數」的指數；每人接觸數的指數是 β − 1（原文：「k ∝ N^(β−1) with β − 1 ≈ 0.12」）。

**Stier 2021**（美國都會區早期 COVID-19 病例成長率；`2026-10-15-d3f52442`；`lit/source/2026-10-15-d3f52442_s42949-021-00030-0.pdf`，出版版本）

> "All of these relationships are tied to socioeconomic networks with average degree (number of social connections per capita, k) that increases approximately as a power law of city size k(N) = k0N^δ, with δ ≈ 1/6, and city size, N, as measured by population size"（正文第 2 段）

> "The observed scaling exponent of COVID-19 case growth of [0.11, 0.20] ~ δ = 1/6 is thus inconsistent with a pure population density effect"（Discussion）

> "By the end of June, the scaling exponent is consistent with 0"（Discussion）

- 注意：量的是**病例成長率**，不是接觸數；1/6 是引用都市縮放理論（Bettencourt 2013、Schläpfer 2014）。
  早期病例成長也可能受各城市檢驗量影響（作者自己列為限制）。

### 1.4 台灣可直接檢驗的資料：2017 台灣社會變遷調查

**TSCS 2017 第七期第三次：網絡與社會資源組**（資料集 `2026-10-15-80a02577`；問卷 `lit/source/2026-10-15-80a02577_C00339_2q.pdf`，報告 `lit/source/2026-10-15-80a02577_C00339_2r.pdf`）

> 「B19.請問您平常一天裡面，從早到晚大概總共跟多少人(含同住家人)接觸？指一對一的接觸，包括講話、見面、打電話、寫信、透過網路，對方是認識或不認識的都算。」（問卷 p. 7；選項 0、1–4、5–9、10–19、20–49、50–99、100 人以上）

> 「B20.(承 B19)請問這些人裡面大概有多少是您平常面對面接觸的?」（問卷 p. 7）

> 「將台灣地區358個鄉鎮市區分為七個層別。在考量地理區因素後，將七個層別調整合併為19個分層，並稱之為地理分層。」（報告 貳、一、（二）抽樣分層；七個層別依人口密度、產業、教育、年齡等八個變項劃分）

- 報告附錄四的 B19 全國次數（n = 1,955）：0 人 5、1–4 人 202、5–9 人 500、10–19 人 604、20–49 人 409、50–99 人 126、100 人以上 103、無法選擇 6（合計 1,955，已核對加總）。
- **限制**：B19 包含電話、網路等非面對面接觸；B20 只問面對面的大約比例。
- 依層別（都市化程度）或地理區交叉分析需要個人資料檔，須以 SRDA 會員身分下載。

### 1.5 小結（我的整理，不是原文）

1. 沒有任何找到的來源支持「每人接觸數與城市人口成正比」（指數 1）。
2. 面對面接觸調查的結果不一致：中國無差異、法國無差異、肯亞鄉村較多、辛巴威鄉村較少、台灣東部較少。
3. 定量縮放只有兩個：Schläpfer 的每人手機聯絡人指數約 0.12（範圍 0.05–0.15，95% CI 0–0.20），以及 Stier 的早期 COVID 病例成長率指數 0.11–0.20（理論值約 1/6）。
   兩者都不是每日面對面接觸數。
5. 台灣可以直接檢驗的資料是 TSCS 2017（B19 × 都市化層別），需取得個人資料檔。
4. **我的推算**（需再確認）：若每人接觸數 ∝ 人口^0.12，人口相差 10 倍的兩地，接觸比約 10^0.12 ≈ 1.32；Fu 2012 的北部或南部對東部約為 12.35 / 9.5 ≈ 1.30。
   兩者數量級相近，但 Fu 2012 沒有定義區域，因此不能直接當成驗證，只能當成假說。

---

## 2. 職場：工作群組與每日職場接觸

### 2.1 模型慣例：工作群組大小固定，不隨公司規模

**Chao 2010（FluTE）**（`2026-10-15-80e052a7`；`lit/metadata/2026-10-15-80e052a7_PMC2813259.xml`）

> "Working individuals are assigned to … a work group of about 20 people to represent their close contacts at the workplace."（Methods；引文中「…」省略的是 "communities and neighborhoods within their destination tracts to simulate casual community contacts during the day, and"）

**Fu 2012 Table 2**（台灣的流感模擬沿用美國 PNAS 模型的 Workgroup 層）

> Table 2 "Workgroup | Adult19–64 | Adult 19–64 | 0.0575 | 5.78% | 1.00 | 0.058"（欄位：PNAS 接觸機率、台灣日記推得的接觸、比例因子、調整後接觸）

**Germann 2006**（PNAS 模型本身；`2026-10-15-81c933db`；`lit/source/2026-10-15-81c933db_germann-et-al-2006-….pdf`，使用者下載）

> "In five of these contexts (households, household clusters, preschools, playgroups, schools, and work groups), relatively close person-to-person association regularly occurs."（Materials and Methods）

> "Because each individual may interact with any member of his or her mixing group, the group sizes determine the numbers of people who would be considered for antiviral prophylaxis in our socially targeted strategy of mitigation (below)."（Materials and Methods）

- 正文只定義「work groups」這個場所，**工作群組的人數寫在 Supporting Text**，正文沒有；Supporting Text 尚未取得。
  目前「約 20 人」只有 FluTE（同一研究群的後續模型）的原文。

### 2.2 職場內實際量到的接觸結構

**Génois 2015**（法國 InVS 辦公室，感測器量測兩週；`2026-10-15-45530f4d`；`lit/source/2026-10-15-45530f4d_genois2015_arxiv.pdf`）

> "…although the study lasted two weeks, the average degree in the aggregated network is only around 15, meaning that each individual met only ≈ 15% of the office population participating to the study over the course of these two weeks."（Section 3.2）

> "…contacts occur much more often inside departments (internal contacts) than between them (external contacts), with the exception of the logistics department."（Section 3.3）

**Potter 2015**（瑞士研究機構；`2026-10-15-3835f9dd`；`lit/metadata/2026-10-15-3835f9dd_PMC4663701.xml`）

> "We found angular distance, workgroup membership, and project collaboration to be highly predictive of contact in this setting."（Conclusion）

- 支持：職場接觸由部門或工作群組決定，同事中只有一小部分會實際碰到。
  CovSyn 中「群組裡只有 45% 會被碰到」在方向上並不違反量測結果。

### 2.3 每日職場接觸數

**Fu 2012 Table 1**（台灣，24 小時日記，每人上限 40 人）

> Table 1 "Location | … | Workplace | 6,033 (24.9)"

> "Our 1,943 contact diaries recorded a total of 24,265 contacted persons during the past 24 hours."

> "About 37.9% of adults were service-sector workers, and 23.8% were manual workers; others were either students (5.5%) or not working (32.8%)."

- **我的推算**：24,265 × 24.9% ÷ 1,943 ≈ 3.1 個職場接觸／人／日（全體受訪者，含兒童與無工作者）。
  換算成「每位就業者」需要受訪者中就業者人數與加權，原文沒有，必須取得原始資料。

**Béraud 2015**（法國）

> "Participants with more than 20 daily professional contacts were asked not to report them but rather to provide their total number and age distribution"（Methods）

> "The specific number of contact made at work was 3[0;10] and 20[6;38] with SPC."（Results；SPC = 超過 20 人時另外回報的工作接觸）

**Mistry 2021**（合成人口，權重以 POLYMOD 校準；`2026-10-15-06dcede1`；`lit/metadata/2026-10-15-06dcede1_PMC7803761.xml`）

> "The regression yields 4.11 contacts (standard error, SE 0.41) in the household setting, 11.41 contacts (SE 0.27) in schools, 8.07 contacts (SE 0.52) in workplaces, and 2.79 contacts (SE 0.48) for the general community setting."（Results）

**Mossong 2008**（POLYMOD，歐洲 8 國；`mossong2008MixingInfectiousPLoSMed`）

> "Of all pooled reported contacts, 23%, 21%, 14%, 3%, and 16% are made at home, at work, at school, while travelling, and during leisure activities, respectively"（Results）

**Danon 2013**（英國；`2026-10-15-adabe851`；`lit/metadata/2026-10-15-adabe851_PMC3712448.xml`）

> "…teachers and service workers have significantly more contacts on working days than the national average, whereas researchers have significantly fewer…"（Results (b)）

### 2.4 依公司規模的職場接觸

- 以 Scopus 與網路搜尋（2026-10-15）**沒有找到**任何量測「每人職場接觸數 vs 公司規模」的研究。
- **TSCS 2017** 同一份問卷有公司規模題，可以和 B19 交叉分析（需個人資料檔）：

> 「c.公司/機構人數(包含本人)：□(01)1-9 人 □(02)10-29 人 □(03)30 人或以上」（問卷 F6c，p. 15）

- 報告附錄四的 F6c 全國次數：1–9 人 767、10–29 人 277、30 人以上 805、不知道 29、拒答 4、跳答 73（合計 1,955）。
- Chen 2022（`2026-10-07-74078dfd`）Table 5 依**行業**列出群聚大小，不是依公司規模（2026-10-12 查證）。

### 2.5 小結（我的整理，不是原文）

1. 「工作群組隨公司規模以 0.27 次方變大」沒有來源支持；文獻中的模型（FluTE、PNAS、Fu 2012 的台灣模擬）都用固定大小的工作群組（約 20 人）。
2. 量測研究顯示職場接觸集中在部門或工作群組內，且每人只碰到同事中的一小部分。
3. 每日職場接觸數：法國中位數 3（四分位距 0–10）、POLYMOD 校準 8.07、台灣 Fu 2012 全體平均約 3.1（我的推算，含非就業者）。
   職業差異很大（Danon 2013、Béraud 2015 的 SPC）。
4. 台灣「每位就業者每天的職場接觸」目前沒有可直接引用的數字。
   最可行的來源是 TSCS 2017 個人資料檔：可算「有工作者 vs 無工作者」的 B19 差異，以及 B19 依公司規模（1–9、10–29、30 人以上）的差異；但 B19 是全部接觸（含電話、網路），不是只有職場。

---

## 3. 2026-10-15 追加：community 長尾與工作群組

### 3.1 Community 接觸的長尾

**Danon 2012**（`2026-10-15-9d121b6f`；`lit/markdown/2026-10-15-9d121b6f_PMC3479920_en.md`）

> However, when groups are taken into account, the distribution is found to possess a lognormal body with a power-law tail with an exponent of −2.45 ± 0.2 for numbers of contacts *K* above 28.（Results）

- 支持：含群體接觸時，每日接觸數的尾巴是冪律，機率密度指數約 2.45（英國）。
- 注意：這是機率密度的指數；我們用 Clauset 方法從疫調 38 筆得到的 1.67 也是機率密度指數，兩者差很多，定義（單日 vs 整段期間、日常接觸 vs 疫調匡列）也不同。

### 3.2 工作群組大小

**Timpka 2016**（`2026-10-15-60ea4232`；`lit/markdown/2026-10-15-60ea4232_timpka2016_en.md`）

> some 60% have physical contact with at least one of the other 20 persons (median) present in their section of the workplace.（Introduction）

> setting the number of interacting workers in each workplace mixing group to the mean number of surrounding co-workers suggested by empirical data (*n* = 20)（Methods；存檔中 `=` 兩側是不換行空格 U+00A0）

> when the subjective workgroup size increases from 1 to >50, the average number of physical contacts per person only increases from around 1 to 2.（Discussion；存檔中 `>` 前有 Markdown 跳脫符號 `\`）

- 支持：工作群組約 20 人（瑞典資料中位數），敏感度分析可用 40 人；**群組變大，每人的身體接觸幾乎不變**，所以不該讓接觸數隨公司規模等比增加。

**Epicast 2.0**（`2026-10-15-c7cbfd23`；`lit/markdown/2026-10-15-c7cbfd23_arxiv2504.03604_en.md`）

> Each industry in each community is subdivided into work-groups, which represent direct co-worker teams – analogous to departments or project teams.（2.4.2）

> Industry and US state specific work-group sizes were derived from the 2019 County Business Patterns survey from the US Census Bureau (CPB)（2.8.3）

> We limited the maximum work-group size to 86 based on studies of workplace contact patterns（2.8.3）

- 支持：用官方工商統計依行業推算工作群組大小，上限 86 人；台灣可用主計總處工商普查照做（**尚未下載**）。

**Potter 2015**（`2026-10-15-73dbc58c`；`lit/markdown/2026-10-15-73dbc58c_arxiv1307.2869_en.md`）

> The three research groups belonged to one institute, and 66 individuals worked for one or more of the three groups.（Data）

- Table 1（Descriptive statistics of 50 members of the Swiss research institute）的 Group 列：「24 Group 1」「19 Group 2」「11 Group 3」。
  轉檔時表格結構遺失，2026-10-15 已依 LaTeX 原檔（`lit/source/2026-10-15-73dbc58c_arxiv1307.2869.tex` 第 312–314 行）在 Markdown 中重建為表格。
- 支持：實測的研究小組 11–24 人，與約 20 人的工作群組同一量級。

### 3.3 台灣資料（資料集，不是引文）

以下數字由 `scripts/summarize_taiwan_contact_data.py` 從原始資料算出，屬**我的計算**，可重跑驗證。

- **Fu 2010 接觸日誌**（`2026-10-15-467602c6`，Zenodo，CC BY 4.0）：community-only（非家、非工作、非學校）每人每日平均 3.23、中位數 2、0 次 40.7%、P90 9、P99 24.5；平日受僱者在工作場所「每天或幾乎每天見面」的接觸平均 4.79、中位數 2、P90 12.5；都市 community 平均 3.12、非都市 3.99。
  每人上限 40 筆。
- **ISSP 2017 台灣**（`2026-10-15-4b81f80d`，GESIS ZA6980）：加權後每日接觸（組中點、100+ 設 150）大城市 24.8、郊區 29.2、小鎮 28.5、鄉村 25.6。
