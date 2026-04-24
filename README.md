# Adult Census Income Dataset — 所得予測分析

## 概要

米国国勢調査データ（[Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)）を用いて、個人の属性情報から年収が $50K を超えるかどうかを予測する二値分類タスクに取り組んだ。  
ロジスティック回帰・ランダムフォレスト・LightGBM の3モデルを比較し、オッズ比・特徴量重要度・SHAP値による多角的な解釈分析を行った。

---

## 使用データ

- **データソース**: UCI Machine Learning Repository — Adult Dataset
- **サンプル数**: 約 48,000 件
- **目的変数**: `income`（`>50K` / `<=50K`）
- **主な特徴量**: age, education, occupation, marital_status, capital_gain, capital_loss, hours_per_week, など

---

## 分析の流れ

```
データ読み込み・前処理
  ↓
特徴量エンジニアリング
  ├─ age → age_bin（~19 / 20~29 / 30~39 / 40~49 / 50~59 / 60~）
  ├─ education → education_level（High / Middle / Low）
  ├─ hours_per_week → hours_bin（Part-time / Full-time / Overtime / Extreme）
  ├─ capital_gain / capital_loss → 対数変換（log1p）
  └─ カテゴリ変数のエンコーディング
  ↓
モデル構築・評価（Logistic Regression / Random Forest / LightGBM）
  ↓
解釈分析（オッズ比 / Feature Importance / SHAP）
  ↓
サブグループ分析（性別ごとのモデル性能）
```

---

## モデル比較

| モデル | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression（statsmodels） | 0.842 | 0.900 |
| Logistic Regression（scikit-learn） | 0.851 | 0.910 |
| Random Forest | 0.863 | 0.918 |
| **LightGBM** | **0.874** | **0.931** |

LightGBM が最も高い精度を記録した。

---

## ロジスティック回帰：オッズ比による解釈

statsmodels を用いたロジスティック回帰モデルのオッズ比（有意な変数のみ）を示す。  
各値は「他の変数を統制した上で、当該カテゴリが基準カテゴリと比べて高収入のオッズが何%変化するか」を表す。

### 主な結果

**資本収益 / 資本損失**
- `capital_gain` が1増加するごとに高収入のオッズは **+8%**
- `capital_loss` が1増加するごとに高収入のオッズは **+4%**

**年齢（基準: 20~29歳）**
- 50~59歳が最も高く **+16%**、次いで40~49歳 **+15%**
- 年齢が上がるほど高収入傾向（ただし60歳以降は +6% に低下）

**学歴（基準: Middle）**
- High（高学歴）: **+16%**
- Low（低学歴）: **−7%**

**職業（基準: Adm-clerical）**
- Exec-managerial: **+13%**（最も高い）
- Prof-specialty: **+9%**
- Farming-fishing: **−10%**（最も低い）

**婚姻状況（基準: Never-married）**
- Married-civ-spouse: **+13%**

**性別（基準: Female）**
- Male: **+5%**

**その他の傾向**
- 週労働時間が多いほど（Overtime / Extreme）高収入オッズが高い
- 自営（法人）はプライベートより高収入傾向、無法人自営は低い傾向
- 米国外出身（中国・ベトナム・南部）は米国出身より低収入傾向

---

## 特徴量重要度（Random Forest）

| 特徴量 | Importance |
|---|---|
| capital_gain | 0.164 |
| education_num | 0.141 |
| marital_status_Married-civ-spouse | 0.124 |
| relationship_Husband | 0.097 |
| age | 0.080 |
| marital_status_Never-married | 0.054 |
| hours_per_week | 0.050 |
| capital_loss | 0.039 |
| occupation_Exec-managerial | 0.034 |
| occupation_Prof-specialty | 0.027 |

---

## SHAP 分析（LightGBM）

SHAP 値を用いてモデルの予測根拠を可視化した。

- **capital_gain**: 投資収入が大きい個人ほど高所得方向へ強く寄与。最も影響力の大きい特徴量。
- **age**: 若いほど低所得方向、年齢が高いほど高所得方向に寄与。
- **education_num**: 学歴が高いほど高所得方向に寄与。
- **capital_loss**: 損失が大きい個人は資産家である可能性が高く、高所得方向に寄与。

### 重要特徴量の総合評価

LightGBM の Feature Importance では `relationship` が最上位であったが、Permutation Importance では `capital_gain` と `age` が上位となった。これは `relationship` と `marital_status` の間の相関構造の影響と考えられる。  
複数の指標を総合すると、所得予測に特に重要な特徴量は以下の通りである。

1. **capital_gain**（投資収入）
2. **age**（年齢）
3. **marital_status**（婚姻状況）
4. **education_num**（学歴年数）

---

## サブグループ分析：性別ごとのモデル性能

| グループ | 高所得予測率 | ROC-AUC |
|---|---|---|
| Male | 約 26% | 0.913 |
| Female | 約 8% | 0.951 |

- 男性の方が高所得と予測される割合が高く、データの実際の所得分布を反映している。
- 一方でモデル性能（ROC-AUC）は女性の方が高い値を示した。これは女性の高所得者が高学歴・特定職種など特徴的な属性を持つことが多く、モデルが識別しやすいためと考えられる。

---

## 使用技術

- **言語**: Python 3.x
- **ライブラリ**: pandas / numpy / scikit-learn / statsmodels / lightgbm / shap / matplotlib / seaborn

---

## ファイル構成

```
.
├── notebook/
│   └── analysis.ipynb       # 分析メインノートブック
├── data/
│   └── adult.csv            # 元データ（Kaggle からダウンロード）
└── README.md
```

---

## 参考

- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [Kaggle: Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)
