# CovSyn — 年齡 Risk Ratio 問題與解法

> 本文件聚焦 CovSyn 模擬中「年齡風險倍率(age risk ratio)」的處理,整理已確認的問題與推薦解法。
> 所有 code 位置以 `檔名:行號` 標示;參數值取自 Firefly 最佳化結果(`firefly_best.txt` 最後一代)。

---

## 0. 背景:Risk Ratio 目前怎麼運作

CovSyn 用 Cheng et al. (2020, *JAMA Intern Med*) 的年齡相對風險,對「二次接觸者」依年齡調整其被感染機率。資料流如下:

| 步驟 | 內容 | 位置 |
|---|---|---|
| ① 取值 | `input_P[63:67]` = `[RR_0-19, RR_20-39, RR_40-59, RR_60+]` | `Data_synthesis_main.py:127` |
| ② 展開 | `np.repeat(RR, [20,20,20,41])` → 0~100 歲逐歲查表 | `Data_synthesis_main.py:128` |
| ③ 抽年齡 | `secondary_contact_age ~ age_p`(全人口分布,**五層共用**) | `Data_synthesize.py:967 / 1013 / 1057 / 1101 / 1145` |
| ④ 乘倍率 | `age_adjusted = RR[age] × base_attack_rate` | `Data_synthesize.py:925` |
| ⑤ 截斷 | `age_adjusted[>1] = 1` | `Data_synthesize.py:927` |
| ⑥ 判定感染 | 用 `age_adjusted` 與亂數比較,決定是否感染 | `Data_synthesize.py:891` |

**關鍵事實**

- `base_attack_rate`(程式中 `adjusted_attack_rate`)是用 Cheng 的 **setting 層、全年齡混合**資料校準出來的(例如 Household 4.6%)。
- 初始值 = Cheng 點估計 `[0.3, 1, 2.19, 1.75]`(0-19 依 1/281 設為 0.3);bounds = Cheng 的 95% CI。
- **Firefly 最佳化後的結果 = `[0.026, 0.138, 0.927, 0.469]`** —— 全部 < 1,方向與文獻相反。

> 註:此年齡處理在 CovSyn **原始版本即如此**(已與 `old/` 原版逐行比對確認),並非後續重構引入的 bug,而是原始設計特性。

---

## 1. 問題清單(依嚴重度排序)

### 問題 1 — 把「相對比值」當「絕對倍率」用,缺正規化 ★最關鍵

**位置:** `Data_synthesize.py:925`

Cheng 的 RR 定義是**相對於 20-39 參考組**:

```
各組攻擊率  a_age = RR_age × a_ref        (a_ref = 20-39 組的攻擊率)
```

但 CovSyn 把 RR 乘在**全年齡平均率 ā**(不是參考組 a_ref)上,而且**沒有再除以平均 RR**:

```
CovSyn 目前:  a_age = RR_age × ā                       ← 錯
正確作法:      a_age = RR_age × ā / Σ(p_age · RR_age)
```

少了分母 `Z = Σ p_age · RR_age`(人口加權平均 RR)。

**後果**

1. **整體攻擊率被平移** —— 調整後的全人口平均 = `ā × Z`。RR 用 Cheng 值時 `Z > 1`,等於把已校準好的 setting 層攻擊率水準無意間放大。
2. **語意失效** —— 這 4 個值不再是「相對 20-39 的比值」,變成隨意相乘的絕對倍率。

---

### 問題 2 — RR 是自由優化參數,卻沒有年齡擬合目標 → 漂成反向

**位置:** cost function,`firefly_optimizer.py:334-431`

成本函數只比對 setting 層(Household / Health care / Others)× 時間 bin,**沒有任何「年齡分層」的擬合項**。因此 4 個 RR 之間的相對大小**完全不受約束**,cost 只在乎它們對 `age_p` 的加權平均(整體水準)。

**後果:** Firefly 沒有誘因重現 Cheng 的 2.19 / 1.75,把 RR 壓成方向相反的 `[0.026, 0.138, 0.927, 0.469]`(40-59 應 > 1 卻為 0.93,60+ 應為 1.75 卻為 0.47)。

---

### 問題 3 — 與 base attack rate 退化(degeneracy)

接續問題 1:「RR 整體偏大 × base 偏小」與「RR 偏小 × base 偏大」對 cost 等價。

**後果:** 最省事的解是把 RR 壓到「平均 ≈ 1」變中性、讓 base attack rate 直接對上 Cheng → 強化問題 2,並製造大量 local optima。

---

### 問題 4 — 參考組(20-39)也被優化,錨點被破壞

**位置:** bounds `parameters_for_initialization.py:906-907`,20-39 的 bounds = `[0, 2]`

Cheng 把 20-39 固定為 RR ≡ 1(參考錨點),但 CovSyn 讓它浮動(best-fit = 0.138)。

**後果:** 「相對於 20-39」這個定義從根本上不成立。

---

### 問題 5 — `>1` 截斷是粗糙的非線性

**位置:** `Data_synthesize.py:927`

`a_age > 1 → 1`。高 RR(2.19)× 較大 base 會觸發截斷,破壞乘法線性關係。目前 base 偏小、較少觸發,屬頂端瑕疵。

---

### 問題 6 — 沒有 per-layer 年齡分布

**位置:** `Data_synthesize.py:967 / 1013 / 1057 / 1101 / 1145`

五層接觸者年齡都從同一個**全人口 `age_p`** 抽樣。School 不偏學齡、Workplace 不偏工作年齡。

**後果:** 即使修好 RR,「層內年齡組成不真實」仍會扭曲年齡效應;且正規化的 `Z` 用全域 `age_p` 也不夠精確(理應 per-layer 計算)。

---

## 2. 推薦解法

### ✅ 方案 A(推薦,最省事且立刻正確)— 固定 RR + 加正規化

**步驟 1:把 RR 鎖成 Cheng 文獻值(不再優化)**

最簡單、不用改 `dim` 與參數索引的做法 —— 在 `parameters_for_initialization.py:906-907` 把 lb = ub 鎖死(如同 `vaccination_rate` 鎖 `[0,0]` 的做法):

```python
age_risk_ratios    = np.array([0.3, 1, 2.19, 1.75])
age_risk_ratios_lb = np.array([0.3, 1, 2.19, 1.75])   # 鎖死
age_risk_ratios_ub = np.array([0.3, 1, 2.19, 1.75])
```

→ 解決問題 2、3、4(方向正確、消除退化、20-39 錨點固定),並少 4 個亂跑維度。

**步驟 2:在乘倍率時加正規化**(解決問題 1)

`Data_synthesize.py:924-927` 改成:

```python
def calculate_age_adjusted_secondary_attack_rate(self, secondary_contact_age, adjusted_attack_rate):
    # Z 可在 __init__ 預先算好(age_p 與 age_risk_ratios 都固定且為 0~100 逐歲)
    # self.Z = np.sum(self.age_p * self.age_risk_ratios)
    age_adjusted_attack_rate = (self.age_risk_ratios[secondary_contact_age] / self.Z) \
                               * adjusted_attack_rate
    age_adjusted_attack_rate[age_adjusted_attack_rate > 1] = 1
    return age_adjusted_attack_rate
```

→ 整體平均維持在校準值 `ā`,同時保留 Cheng 的年齡相對結構。

**重跑 Firefly:** 需要(bounds 改變了)。

---

### 方案 B(完整,但工程量大)— 加年齡 cost 項,讓 RR 真的被校

若堅持要「學」這 4 個值,必須:

1. 用 Cheng 年齡欄(contacts `281 / 1161 / 794 / 331`、attack rate `0 / 0.5 / 1.1 / 0.9 %`)做成 ground-truth;
2. 在 cost function 加一個**年齡分層的 attack-rate SSE 項**(類似現有的 setting 層 cost);
3. 仍建議固定 20-39 = 1(保留錨點)。

→ 解決問題 2,但問題 1 的正規化、問題 6 的 per-layer 年齡仍須一併處理。**僅在特別需要「可學的年齡效應」時才值得。**

---

### 方案 C(進階,可選)— 補 per-layer 年齡分布(解決問題 6)

用既有資料推導各層接觸者年齡,取代全域 `age_p`:

- School ∝ `age_p × student_p[age]`
- Workplace ∝ `age_p × employment_p[age]`
- Household 用 `family_size_dict` 的家庭結構
- 並把正規化 `Z` 改成**逐層**計算:`Z_layer = Σ p_age^layer · RR_age`

→ 屬「模型精緻化」。建議在方案 A 落地、確認方向正確後再做。

---

## 3. 推薦執行順序

```
1. 方案 A 步驟 1 + 2(鎖 RR = 文獻值 + 加正規化)   ← 一次解決問題 1~4,CP 值最高
2. 重跑 Firefly,確認 setting 層攻擊率仍擬合良好、年齡方向恢復正確
3. (可選) 方案 C:補 per-layer 年齡分布
4. (僅在需要可學年齡效應時) 方案 B:加年齡 cost 項
```

---

## 4. 需要先決定的事

| 決策 | 影響 |
|---|---|
| **RR 要「固定」還是「可學」?** | 決定走方案 A(固定)或 B(可學)。**推薦固定** —— 在沒有年齡資料進 cost 之前,可學沒有意義。 |
| **base attack rate 代表「全年齡平均」還是「20-39 參考組」?** | 決定正規化公式。預設校準對象是 Cheng setting 層全年齡值 → 採方案 A 的 `/Z` 正規化。 |
| **是否需要合成資料的「被感染者年齡分布」符合現實?** | 若需要 → 一定要做方案 A(+C);若只在乎 setting 層擬合 → 影響較小。 |

---

## 附錄:關鍵數學

設:

- `a_ref` = 20-39 參考組攻擊率
- `RR_age` = Cheng 年齡相對風險(`RR_20-39 ≡ 1`)
- `p_age` = 人口年齡分布
- `ā` = 全年齡平均攻擊率(= CovSyn 校準的 base attack rate)

則:

```
全年齡平均       ā = Σ p_age · a_age = a_ref · Σ p_age · RR_age = a_ref · Z
正確的各組率     a_age = RR_age · a_ref = RR_age · ā / Z          其中 Z = Σ p_age · RR_age
CovSyn 目前算成   a_age = RR_age · ā    (缺了 1/Z 正規化)
```

當 `RR` 取 Cheng 值時 `Z > 1`,故目前實作會把整體攻擊率放大 `Z` 倍,並使 4 個 RR 與 base attack rate 退化。
