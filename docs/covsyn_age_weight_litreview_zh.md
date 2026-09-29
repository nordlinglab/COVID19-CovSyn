# 用有原則的年齡權重取代 Age Risk Ratio — 概念 + 文獻回顧

> 目標:停止把 Cheng 的年齡 **risk ratio** 當成每日 SAR 的自由乘數
> (`Data_synthesize.py:925`,見 [covsyn_risk_ratio_issues.md](covsyn_risk_ratio_issues.md)),改用一個
> **有文獻根據、定義清楚、有外部界限**的年齡權重。

---

## 1. 關鍵概念洞見:把一個權重拆成兩個

Cheng 的「年齡別**二次臨床攻擊率 (secondary clinical attack rate)** risk ratio」其實綁了**兩個不同的年齡效應**:

```text
臨床攻擊率(age)  =  易感性(age)          ×  臨床比例(age)
                   「會不會被感染?」          「感染後會不會出症狀?」
```

- **易感性 (susceptibility)** 在兒童較低、成人大致持平 —— 它屬於**每接觸的感染機率**(正是 `r_a` 現在的位置)。
- **臨床比例 (clinical fraction)** 隨年齡單調上升 —— 它屬於**有症狀/無症狀轉移**,CovSyn *本來就分開建模*
  (`infection_to_recovered_transition_p`,`Data_synthesize.py:350`),只是目前與年齡無關。

**為什麼 `r_a` 看起來隨年齡單調上升 (0.3 → 2.19):** 因為它是「臨床」攻擊率比值 = 易感性 × 臨床比例。
兒童:低 × 低 → 非常低。老人:約持平 × 高 → 高。把兩者烤進同一個 SAR 乘數,正是混淆的根源。
有原則的修法是把它們**拆開**:

| 取代 `r_a` 的 | 在 code 裡用在哪 | 文獻依據 |
|---|---|---|
| **σ(age)** 年齡別*易感性*權重 | 每接觸感染機率(`calculate_age_adjusted_secondary_attack_rate`,L925) | Zhang 2020、Davies 2020、Viner 2021、Franco 2022 |
| **p_clin(age)** 年齡別*臨床比例* | 症狀發作轉移(L350-352) | Davies 2020 |

若你要**最小改動**(保留 CovSyn「臨床 SAR」定義、只用一個權重),那就保留單一權重
`w(age) = σ(age)·p_clin(age)` 乘在 SAR 上 —— 但要**固定成文獻值**、正規化,不要自由 fit。完整拆開比較乾淨,為推薦做法。

---

## 2. 文獻回顧 —— SARS-CoV-2 感染的年齡別易感性

所有數值皆為 **given 接觸下的相對易感性**(該乘在每接觸機率上的量)。各研究的參考組不同,逐列標註。

| 研究 | 設計 | 年齡組 | 相對易感性 | 95% CI | 參考組 |
|---|---|---|---|---|---|
| **Zhang et al. 2020, *Science*** | 湖南接觸追蹤 + RT-PCR(易感性黃金標準) | <15 | **0.34** | 0.24–0.49 | 15–64 = 1 |
| | | 15–64 | 1.00 | — | (參考) |
| | | >65 | **1.47** | 1.12–1.92 | 15–64 = 1 |
| **Davies et al. 2020, *Nat. Med.*** | 年齡結構模型擬合,6 國 | 0–9 | **0.40** | 0.25–0.57 | 成人 |
| | | 60–69 | 0.88 | 0.70–0.99 | 成人 |
| | | <20(摘要) | ≈ 0.5 | — | ≈ 成人的一半 |
| **Viner et al. 2021, *JAMA Pediatr.*** | meta-analysis,32 篇 | <20 vs ≥20 | **0.56**(odds) | 0.37–0.85 | ≥20 = 1 |
| **Franco et al. 2022, *PLOS Comput. Biol.*** (比利時) | 系列血清 + 接觸資料擬合 | [0,6) | **0.182** | 0.146–0.230 | [18,30) = 1 |
| | | [6,12) | 0.550 | 0.427–0.629 | [18,30) = 1 |
| | | [12,18) | 0.603 | 0.536–0.700 | [18,30) = 1 |
| | | [18,30) | 1.00 | 0.83–1.25 | (參考) |
| | | 較年長成人 | 「略為下降」 | — | [18,30) = 1 |

**共識:** 兒童明顯較不易感(≈0.2–0.6,<10 歲最低)。成人 ≈1。
**老人有分歧:** Zhang(接觸追蹤)發現 >65 **較高**(1.47);Davies/Franco(模型擬合)約為**持平或略低**。
→ 60+ 權重應給**較寬的界限**。

## 2b. 較新論文 (2023–2024) 與一個關鍵的變異株警告

針對較新 / 台灣論文重新搜尋,結果如下。

**Uthman et al. 2024, *PLoS One* — 家戶接觸 meta-analysis,按變異株分層的易感性**(最有用的*新*來源;
為**正確的時代**解決了老人/變異株的不確定性):

| 變異株 | 年齡組 vs 成人 20+ | 易感性 OR | 95% CI |
|---|---|---|---|
| **Wild-type**(= CovSyn 台灣 2020 情境) | 0–9 | 0.72 | 0.49–1.05 |
| | 10–19 | 0.77 | 0.68–0.88 |
| | **0–19 合併** | **0.58** | **0.44–0.77** |
| Alpha | 0–19 | 1.10 | 0.93–1.30 |
| Delta | 0–19 | 0.91 | 0.86–0.96 |

→ **Wild-type 0–19 ≈ 0.58 (0.44–0.77)** 與 Viner (0.56)、Zhang (0.34) 高度一致 —— 所以 0–19 易感性權重在
2020 時期穩穩錨定在 **~0.5**。

**⚠ 關鍵警告 —— 年齡易感性是變異株特異的。** 兒童的易感性缺口在較新變異株中**縮小甚至反轉**:
Alpha 0–19 OR ≈ 1.10,而 Omicron 時代研究(如日本 Toyama,*CDC EID* 2023;下列台灣 2022–2023 世代)顯示
兒童在家戶中變得**高度**易感(Omicron 家戶 OR 最高達 pre-VOC 的 ~6 倍)。**CovSyn 模擬的是台灣 2020 =
wild-type,所以要用 wild-type 值(兒童 ~0.5),不要用 Omicron 時代的值。**

**找到的較新台灣論文 —— 但都是 Omicron 時代(對 CovSyn 2020 是錯的變異株):**
- *Epidemiological characteristics of the three waves of COVID-19 in Taiwan, Apr 2022–Mar 2023*(BA.2/BA.5/BA.2.75)。
- *台灣家戶世代 (May 2022–Nov 2023)*:整體家戶 SAR **37.2%**(Omicron),接種 index 較低(34.9% vs 63.2%)、
  有感染史 index 更低(27.0% vs 46.3%)。

  這些是 **Omicron**,被疫苗/免疫效應主導,**不適合**用於 2020 wild-type 的年齡權重。只有在 CovSyn 之後
  擴展到 Omicron 情境時才有用。

**新搜尋的結論:** 除了 Cheng 2020(§4c)之外,沒有新的*台灣、wild-type、年齡分層易感性*論文。最佳的**新**
輸入是按變異株分層的 **Uthman 2024** meta-analysis,它(對 wild-type)把 0–19 權重收緊到 ~0.5 並確認 §4a。
建議不變;0–19 界限可用 Zhang(下界)和 Uthman wild-type(上界)收緊到 **[0.34, 0.77]**。

## 2c. 跨國異質性 —— 各國差很多嗎?

**形狀:各國一致。水準:差很多,但主要不是因為國家。**

- **Madewell 2020(家戶 meta,54 研究,77,758 人):** 成人接觸 **28.3%**(20.2–37.1)vs 兒童接觸 **16.8%**
  (12.3–21.7)→ 成人約為兒童的 **1.7 倍**(兒童:成人 ≈ 0.59)。**中國(21 研究)與其他國家(33)之間無顯著差異。**
- 高異質性(Madewell I² = 96.8% 成人 / 78.9% 兒童;Viner I² = 94.6%)是由**變異株、檢測強度、病例定義
  (臨床 vs PCR)、接觸場域、家戶大小**驅動,而非國家本身。

兒童:成人易感性比值在各種不同情境下聚集於 **~0.34–0.59**(Zhang 0.34、Davies 0.40、Viner 0.56、
Uthman wild-type 0.58、Madewell 0.59)→ **年齡形狀可移植**;只有絕對水準不同。

---

## 2d. 整合後的 WILD-TYPE 年齡分布(與 CovSyn = 台灣 2020 相關的那一個)

兩種不同的「年齡分布」—— 要分開看:

### A. 感染的相對**易感性**(wild-type),參考 20–39 = 1

| 來源(國家) | 0–19 | 20–39 | 40–59 | 60+ |
|---|---|---|---|---|
| Zhang 2020(中國) | 0.34 (0.24–0.49)¹ | 1(參考 15–64) | 1 | 1.47 (1.12–1.92)² |
| Davies 2020(6 國) | ≈0.40–0.50(0–9: 0.40) | ~1 | ~0.9 | 0.88 (0.70–0.99)³ |
| Viner 2021(32 研究 meta) | 0.56 (0.37–0.85) | 1 | 1 | — |
| Uthman 2024(家戶 meta) | 0.58 (0.44–0.77) | 1 | 1 | — |
| Franco 2022(比利時) | 0.18–0.60⁴ | 1(參考 18–30) | ~0.9 | 略 <1 |
| Madewell 2020(家戶 meta) | 兒童:成人 ≈0.59 | 1 | 1 | — |
| **整合(建議)** | **≈0.50** (0.34–0.77) | **1.00**(錨點) | **≈0.95** (0.80–1.10) | **≈1.10** (0.70–1.90,不確定) |

¹ <15 vs 15–64 · ² >65 vs 15–64 · ³ 60–69 · ⁴ 組內梯度強:<6 ≈0.18,12–18 ≈0.60

**形狀:** 兒童約成人的一半(梯度強:幼兒 ≈0.2,青少年 ≈0.6);成人持平 ≈1;老人**不確定**
(中國↑ 1.47 vs 模型擬合 ≈0.9)。

### B. **臨床(有症狀)攻擊率**比值(wild-type)—— Cheng/台灣量的就是這個

這 = 易感性 × 臨床比例,所以隨年齡爬升更陡(老人有症狀比例較高)。參考 20–39 = 1。

| 來源 | 0–19 | 20–39 | 40–59 | 60+ |
|---|---|---|---|---|
| **Cheng 2020(台灣),臨床 SAR %** | 0% (0/281) | 0.5% | 1.1% | 0.9% |
| **Cheng 相對 20–39 比值** | ~0(floor 0.3) | 1 | **2.2** | **1.8** |
| 由易感性×臨床比例推得 | 低×低 → 非常低 | 1 | ↑ | ↑↑ |

→ 這正是 CovSyn 現在的 `[0.3, 1, 2.19, 1.75]`。它是**臨床**比值(不是易感性),所以才隨年齡上升,也正是
為什麼把它拆開(§1)是乾淨的修法。

### C. 年齡別**臨床比例**(wild-type,Davies)—— 第二塊

| 0–19 | 20–39 | 40–59 | 60+ |
|---|---|---|---|
| ~0.21 (0.12–0.31) | ~0.35 | ~0.50 | ~0.69 (0.57–0.82) |

**兜起來(wild-type):** 臨床_SAR_比值(age) ≈ 易感性(age) × 臨床比例(age) / (20–39 的值)。
- 0–19: 0.50 × 0.21 → 非常低 ✓(台灣 0)
- 40–59: 0.95 × 0.50 → 偏高 ✓(台灣 2.2)
- 60+: 1.10 × 0.69 → 偏高 ✓(台灣 1.8)

這個分解重現了台灣的臨床模式 —— 證明 §1 的拆解在數學上自洽。

---

## 2e. 不把任何單一年齡組設成 1 的易感性

**先講方法學現實:** 年齡易感性*本質上是相對的* —— 你無法直接量到「given 一次接觸的絕對易感性」,因為
每個估計都需要一個尺度。尺度只能是:(a) 某個年齡組設成 1,或 (b) **人口平均**設成 1,或退回 (c) 某個
**絕對可觀測量**(per-contact SAR 或血清陽性率),但後者混了易感性與接觸/暴露。所以「沒有年齡 = 1」其實
是指 **選項 (b) 或 (c)**。

### 選項 (b) —— 正規化成人口平均 = 1(沒有任何單一年齡被釘住)
- **OpenABM-Covid19 (Hinch et al. 2021)** —— `relative_susceptibility` 明確「正規化成人口中個體的平均
  易感性 = 1」。值 [0.35, 0.69, 1.03×4, 1.27, 1.52, 1.52],沒有單一年齡 = 1。
  → **這正是 CovSyn 計畫裡的 `/Z` 正規化。**

> **實作關鍵:** `ã_l^a = (σ[g_c]/Z)·ã_l`(Z = Σ age_p·σ)讓**人口平均 = 1,而非任何單一年齡**。所以這個
> 計畫已經滿足你「沒有年齡 = 1」的偏好 —— 它走的是 OpenABM 慣例,不是「20–39 = 1」慣例。

### 選項 (c) —— 絕對年齡別感染率(血清 / per-contact),完全無參考組
這些報告原始的年齡別感染 %(你再自己推相對形狀):

- **Pollán et al. 2020, *Lancet*(ENE-COVID,西班牙,n≈61,000)** —— 血清陽性 0–19 歲 **3.4–3.8%** vs
  成人 **4.4–6.0%**;<10 歲 **<3.1%**。絕對、全國。
- **Stringhini et al. 2020, *Lancet*(日內瓦血清調查)** —— 血清陽性 **0.8%**(5–9 歲)vs **9.6%**
  (10–19)vs **9.9%**(20–49)。絕對、分年齡。
- **Jing et al. 2020, *Lancet Infect Dis*(廣州家戶)** —— 絕對家戶 SAR **5.2%**(<20)、**14.8%**
  (20–59)、**18.4%**(≥60)。

⚠ 血清/SAR 的絕對數字混入了**暴露與接觸模式**,所以不是純易感性 —— 拿來當**形狀證據 / 驗證**,不要直接
當每接觸權重。

### 找到的較新易感性參數論文(供完整性)
- **Boldea et al. 2024, *PNAS Nexus*** —— 按**變異株**分層的年齡別易感性(荷蘭);兒童在 Omicron 之前
  遠低於成人,之後上升。(錨定成人 >19 = 1;按變異株分層,類似 Uthman 2024。)
- **Hu et al. 2021, *Nature Communications*(湖南,1178 感染者 / 15,648 接觸者)** —— 扣接觸後易感性
  **隨年齡上升**;傳染力各年齡無差。
- **Goldstein, Lipsitch & Cevik 2021, *J Infect Dis*** —— q-susceptibility 綜述:兒童約為成人易感性的
  **20–50%**。

**結論:** 沒有任何論文給出「完全無尺度的絕對易感性」(定義上不可能)。乾淨的「沒有單一年齡 = 1」選擇是
**人口平均正規化(OpenABM)** —— 這正是 CovSyn `/Z` 計畫在做的。絕對血清/SAR 論文(Pollán、Stringhini、
Jing)是最好的「無參考年齡」*觀測*證據,用於形狀/驗證。

## 3. 文獻 —— 年齡別臨床(有症狀)比例

用於**症狀發作轉移**,不是 SAR 乘數。

| 研究 | 年齡組 | 臨床比例 | 95% CI |
|---|---|---|---|
| **Davies et al. 2020, *Nat. Med.*** | 10–19 | **0.21** | 0.12–0.31 |
| | (隨年齡單調上升) | | |
| | 70+ | **0.69** | 0.57–0.82 |

形狀:有症狀比例**隨年齡單調上升**,~0.2(兒童)→ ~0.7(老人)。

---

## 4. 建議的 CovSyn 權重與 Firefly 界限(4 組:0–19, 20–39, 40–59, 60+)

參考組 = **20–39(鎖在 1)**。界限取上述 CI 的聯集,老人因跨研究分歧而稍微放寬。

### 4a. 易感性權重 σ(age) —— 取代感染機率上的 `r_a`

| 年齡組 | 初值 | 下界 | 上界 | 理由 |
|---|---|---|---|---|
| 0–19 | **0.45** | 0.34 | 0.77 | Zhang 0.34、Viner 0.56、**Uthman 2024 wild-type 0.58 (0.44–0.77)** |
| 20–39 | **1.00** | 1.00 | 1.00 | 參考,**鎖死** |
| 40–59 | **0.95** | 0.70 | 1.15 | 成人 ≈1,略降(Davies/Franco) |
| 60+ | **1.10** | 0.70 | 1.92 | Zhang 1.47(–1.92) vs Davies 0.88 → **寬** |

> 採用 **mean-preserving** 使其不會平移已校準的 base SAR:
> `σ_norm(age) = σ(age) / Z`,`Z = Σ_age p_age·σ(age)`(人口加權平均)。這就是
> [covsyn_risk_ratio_issues.md](covsyn_risk_ratio_issues.md) 缺的 `/Z`,現在套在一個意義明確
> (「相對 20–39」)的*易感性*權重上。

### 4b. 臨床比例 p_clin(age) —— 給症狀發作轉移用(選用,完整版修法)

| 年齡組 | 初值 | 下界 | 上界 | 來源 |
|---|---|---|---|---|
| 0–19 | 0.21 | 0.12 | 0.35 | Davies 10–19 = 0.21 (0.12–0.31) |
| 20–39 | 0.35 | 0.25 | 0.50 | Davies 內插 |
| 40–59 | 0.50 | 0.40 | 0.62 | Davies 內插 |
| 60+ | 0.69 | 0.57 | 0.82 | Davies 70+ = 0.69 |

這會讓 `infection_to_recovered_transition_p`(= P(無症狀))變成年齡相依 = `1 − p_clin(age)`,取代現在的單一值。

---

## 4c. 台灣專屬資料(CovSyn 本來就在用的來源)

台灣**有**年齡分層資料 —— 就是 Cheng et al. 2020,CovSyn 校準用的那篇。但它量的是**臨床攻擊率**
(易感性 × 臨床比例),不是純易感性,而且樣本很小。

**Cheng et al. 2020, *JAMA Intern. Med.*(台灣,Table 2)—— 按接觸者年齡的二次臨床攻擊率:**

| 年齡組 | 接觸數 | 二次病例 | 臨床攻擊率 (95% CI) | 相對 20–39 比值 |
|---|---|---|---|---|
| 0–19 | 281 | 1(無症狀) | **0%** (0–1.4%) | 0 → CovSyn floor 0.3 |
| 20–39 | 1,161 | 8 | **0.5%** (0.2–1.1%) | 1(參考) |
| 40–59 | 794 | 10 | **1.1%** (0.6–2.1%) | **2.2** → CovSyn 2.19 |
| 60+ | 331 | 3 | **0.9%** (0.3–2.6%) | **1.8** → CovSyn 1.75 |
| 全體 | 2,761 | 22 | 0.7% (0.4–1.0%) | — |
| 家戶 | 151 | — | 4.6% (2.3–9.3%) | — |

→ 這**正是** CovSyn `[0.3, 1, 2.19, 1.75]` 初值與 Table S3 界限的來源(Cheng 95% CI 換算成比值;
0–19 設成 0.3 = 1/281,因為它有 **0 個臨床**病例)。

**Huang, Tu & Lai 2021, *J. Microbiol. Immunol. Infect.*(台灣全國 meta-analysis):** 整體 SAR
0.84%(95% CrI 0.42–1.69%)—— **無年齡分層**,只能定整體水準。

**為什麼台灣資料不足以定 4 個年齡權重:**
- 全體只有 **22 個二次病例**;0–19 是 **0 個臨床**病例(0.3 是 floor 不是估計值);60+ 只靠 **3 個**。
  CI 大到爆 → 最佳化器幾乎不受約束 → 漂成 `[0.026, 0.138, 0.927, 0.469]`。
- 它是**臨床**比值,所以隨年齡上升一部分反映的是**臨床比例**而非易感性 —— 與 §1 的拆解一致。

**台灣 vs 國際資料的建議用法:**
- 用 **Cheng(台灣)** 當年齡別*臨床*攻擊率的**水準 / 驗證目標**(本地 ground truth,也是 CovSyn cost 用的資料)。
- 年齡**形狀**(易感性 σ、臨床比例 p_clin)借**較大型國際研究**(Zhang/Davies/Viner/Franco,數千事件)當
  **informative prior / 界限**,因為台灣單獨太稀疏,撐不起 4 個數字。
- 具體:用 §4a 的 σ(age)(國際形狀)固定/正規化,用 §4b 的 p_clin(age)(Davies),最後用 Cheng 的
  `0 / 0.5 / 1.1 / 0.9 %` **驗證**模擬出的年齡別臨床攻擊率。

## 5. 這如何修好原本的問題

| 原本問題(risk_ratio_issues) | 由什麼修好 |
|---|---|
| 相對比值當絕對乘數、缺 `/Z` | σ 明確為「相對 20–39」**且**以 `Z` 正規化(§4a) |
| 自由參數方向漂錯 | σ **固定/界限取文獻 CI**;20–39 鎖為錨點 |
| 參考組沒鎖 | 強制 20–39 = 1 |
| 臨床 vs 易感性混淆 | 拆成 σ(感染)與 p_clin(症狀) |
| 沒有年齡分層驗證 | σ 與 p_clin 變成可從 output 計算的**驗證目標**(模擬易感性比值、模擬年齡別臨床比例) |

---

## 6. 實作指引

- **易感性權重:** 把 `age_risk_ratios` 改名/改用為 `age_susceptibility`,並把
  `calculate_age_adjusted_secondary_attack_rate`(`Data_synthesize.py:924-927`)改成
  `ã_l^a = min(1, (σ[g_c]/Z) · ã_l)`;在 `__init__` 預先算 `Z = Σ age_p·σ`。
- **界限:** 在 `parameters_for_initialization.py:935-940` 依 §4a 設定(鎖 20–39 lb=ub=1)。
- **臨床比例(選用):** 在 `draw_course_of_disease`(`Data_synthesize.py:350`)用 `secondary_contact_age`/
  個案年齡讓 `transition_p[0]` 變成年齡相依。
- **重跑 Firefly**(界限變了)。若 σ 全鎖死,少 4 個自由維度;若保留可學,要加年齡分層 SAR cost 項(否則又漂)。

---

## 7. 注意事項 / 待決問題

- **易感性 vs 傳染力:** 這裡的權重是**感染者端易感性** —— 對應 `r_a` 的位置正確。感染者端傳染力較不確定
  (Franco:「CI 非常大」),且已由 day-since-infection 的 SAR 曲線處理;沒有資料前不要再加第二個年齡傳染力項。
- **老人易感性確實不確定**(Zhang↑ vs Davies/Franco 持平)。60+ 界限保持寬,或把 60+ 當唯一可學參數並加年齡 cost。
- **CovSyn base SAR 的定義:** 若它仍是*臨床*(有症狀)攻擊率、且你**不**拆出 p_clin,那就用合成權重
  `w(age)=σ·p_clin` 而非純 σ —— 但要固定,不要自由 fit。
- **per-layer 年齡分布:** σ 是在接觸者年齡 `g_c ~ age_p`(全域)抽樣。School/workplace 年齡組成不同;
  `Z` 理想上應逐層計算(見 [covsyn_risk_ratio_issues.md](covsyn_risk_ratio_issues.md) 問題 6)。

---

## 來源

- [Zhang et al. 2020, *Science* — Changes in contact patterns shape the dynamics of the COVID-19 outbreak in China](https://www.science.org/doi/10.1126/science.abb8001)(易感性 <15: 0.34 [0.24–0.49];>65: 1.47 [1.12–1.92])
- [Davies et al. 2020, *Nature Medicine* — Age-dependent effects in the transmission and control of COVID-19 epidemics](https://www.nature.com/articles/s41591-020-0962-9)(易感性 0–9: 0.40 [0.25–0.57];臨床比例 10–19: 0.21 → 70+: 0.69)
- [Viner et al. 2021, *JAMA Pediatrics* — Susceptibility to SARS-CoV-2 Infection Among Children and Adolescents Compared With Adults](https://pmc.ncbi.nlm.nih.gov/articles/PMC7519436/)(<20 vs ≥20 odds 0.56 [0.37–0.85])
- [Franco et al. 2022, *PLOS Comput. Biol.* — Inferring age-specific differences in susceptibility to and infectiousness upon SARS-CoV-2 infection (Belgian data)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009965)(參考 [18,30)=1;[0,6): 0.182;[6,12): 0.550;[12,18): 0.603)

### 不把任何年齡錨定為 1 的易感性(人口平均 / 絕對)
- [Hinch et al. 2021, *PLOS Comput Biol* — OpenABM-Covid19(relative_susceptibility 正規化成人口平均 1)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009146);[參數文件](https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/infection_parameters.md)
- [Pollán et al. 2020, *Lancet* — ENE-COVID 全國血清盛行率研究,西班牙](https://www.sciencedirect.com/science/article/pii/S0140673620314835)(絕對年齡別血清陽性:0–19 歲 3.4–3.8%)
- [Stringhini et al. 2020, *Lancet* — 日內瓦血清調查](https://pmc.ncbi.nlm.nih.gov/articles/PMC7546669/)(5–9 歲 0.8% vs 20–49 歲 9.9%)
- [Boldea et al. 2024, *PNAS Nexus* — Age-specific transmission dynamics over first 2 years(按變異株分層的易感性)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10837015/)
- [Hu et al. 2021, *Nature Communications* — Infectivity, susceptibility, and risk factors under intensive contact tracing in Hunan](https://www.nature.com/articles/s41467-021-21710-6)(易感性隨年齡上升)
- [Goldstein, Lipsitch & Cevik 2021, *J Infect Dis* — On the effect of age on transmission(q-susceptibility)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7386533/)

### 較新 (2023–2024)
- [Uthman et al. 2024, *PLoS One* — Susceptibility and infectiousness of SARS-CoV-2 in children versus adults, by variant (wild-type, Alpha, Delta): household-contact meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11379298/)(wild-type 0–19 易感性 OR 0.58 [0.44–0.77];年齡效應變異株特異)
- [Variants & Age-Dependent Infection Rates among Household and Nonhousehold Contacts, *CDC EID* 2023 (日本 Toyama)](https://wwwnc.cdc.gov/eid/article/29/8/22-1582_article)(Omicron:兒童家戶感染升到 ~38%;年齡模式隨變異株翻轉)
- [Epidemiological characteristics of three COVID-19 waves in Taiwan, Apr 2022–Mar 2023 (Omicron)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10213295/)
- [Taiwan household cohort, vaccinated index cases & transmission (May 2022–Nov 2023, Omicron)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975059/)(家戶 SAR 37.2%)

### 台灣專屬 (2020, wild-type —— 與 CovSyn 相符)
- [Cheng et al. 2020, *JAMA Internal Medicine* — Contact Tracing Assessment of COVID-19 Transmission Dynamics in Taiwan](https://pmc.ncbi.nlm.nih.gov/articles/PMC7195694/)(Table 2 年齡分層臨床 SAR:0–19: 0%;20–39: 0.5%;40–59: 1.1%;60+: 0.9%;整體 0.7%;家戶 4.6%)—— **CovSyn `[0.3,1,2.19,1.75]` 的來源**
- [Huang, Tu & Lai 2021, *J. Microbiol. Immunol. Infect.* — Estimation of the secondary attack rate of COVID-19 using nationwide contact-tracing data in Taiwan](https://pmc.ncbi.nlm.nih.gov/articles/PMC7289119/)(整體 SAR 0.84%,95% CrI 0.42–1.69%;無年齡分層)
