# Predict-Future-Sales-DNSC-3288-Project

## 1. Basic Information

- **Course:** DNSC 3288 — Big Data, Predictive Analytics, & Ethics
- **Semester:** Fall 2025 
- **Project:** Kaggle *Predict Future Sales* Competition
- **Author(s):**
  - Alexis F, alexis@gwu.edu 
- **File with model implementation:**
  - [`Alexis_DNSC_3288_Final_Project.ipynb`](./Alexis_DNSC_3288_Final_Project.ipynb)
- **License:** Apache 2.0 
- **Model type:** Gradient-boosted decision tree regressor (`XGBRegressor` from XGBoost)

### Intended Use

- **Intended purpose:**  
  Predict **monthly item sales** (`item_cnt_month`) for each combination of `shop_id` and `item_id` one month into the future, to support retail decision-making and forecasting.

- **Intended users:**
  - Students learning machine learning/time-series forecasting.
  - Data analysts experimenting with the Kaggle *Predict Future Sales* dataset.

- **Out-of-scope / inappropriate uses:**
  - Any use beyond educational purposes is considered out-of-scope.

---
## 2. Training Data

### Source of Training Data
The training data comes from the public Kaggle competition:  
**Predict Future Sales** (Russia, 2013–2015)  
(https://www.kaggle.com/c/competitive-data-science-predict-future-sales)

### How Training Data Was Divided
A time-based split ensures realistic forecasting:

| Dataset Split | Months Used | Purpose |
|--------------|------------|---------|
| Training | 0–32 | Model fitting |
| Validation | 33 | Held-out evaluation |
| Test | 34 | Kaggle submission (target unknown) |

### Number of Rows

| Split | Rows | Notes |
|-------|------|------|
| Training | **17,388,360** | 15 modeled input features |
| Validation | **526,920** | 15 modeled input features |

These sizes reflect the **full grid** of all shop–item combinations across months, including months with zero sales.

---

### Data Dictionary

| Feature                   | Role   | Type     | Description |
|--------------------------|--------|---------|-------------|
| date_block_num           | Input  | int8    | Month index (0 = Jan 2013 … 33 = Oct 2015) |
| shop_id                  | Input  | int16   | Shop identifier |
| item_id                  | Input  | int32   | Item identifier |
| item_cnt_month           | Target | float32 | Monthly sales count for that shop–item |
| month                    | Input  | int8    | Month extracted from `date_block_num` |
| year                     | Input  | int8    | Year extracted from `date_block_num` |
| item_cnt_month_lag_1     | Input  | float32 | Sales last month |
| item_cnt_month_lag_2     | Input  | float32 | Sales 2 months ago |
| item_cnt_month_lag_3     | Input  | float32 | Sales 3 months ago |
| item_cnt_month_lag_6     | Input  | float32 | Sales 6 months ago |
| item_cnt_month_lag_12    | Input  | float32 | Sales 12 months ago |
| item_month_avg_lag_1     | Input  | float32 | Average item sales across shops 1 month ago |
| item_month_avg_lag_2     | Input  | float32 | Average 2 months ago |
| item_month_avg_lag_3     | Input  | float32 | Average 3 months ago |
| item_month_avg_lag_6     | Input  | float32 | Average 6 months ago |
| item_month_avg_lag_12    | Input  | float32 | Average 12 months ago |

---
## 3. Test Data

- **Rows:** 214,200 (from Kaggle `test.csv`)
- **Contains:**
  - `ID`, `shop_id`, `item_id`
- **Unknown target:** `item_cnt_month`
- Treated as **month 34** for prediction

### Train–Validation Split

| Split | Months | Purpose |
|-------|--------|---------|
| Train | 0–32   | Model training |
| Validation | 33 | Held-out evaluation |
| Test (Kaggle) | 34 | Prediction submission |

---
## 4. Model Details

### Input & Target
- **Target:** `item_cnt_month` (clipped to [0, 20])
- **Inputs:** All features except the target (see Data Dictionary)

### Model Type
- Algorithm: Gradient Boosted Decision Trees for regression
- Library: XGBRegressor from the XGBoost Python package

### Software Environment
- Language: Python 3.10
- Key Packages: xgboost, pandas, numpy, scikit-learn, matplotlib
  

### Hyperparameters
```python
XGBRegressor(
    max_depth=10,
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    eval_metric='rmse',
    n_jobs=-1,
    random_state=42
)
```

---

## 5. Model Evaluation

### Metrics Used
- The project uses three evaluation metrics:

| Metric | Type | Why Included |
|--------|------|--------------|
| RMSE (Root Mean Squared Error) | Regression | Kaggle competition metric |


---
### 5.1 Performance Summary

| Dataset | Metric | Value |
|---------|--------|-------|
| Validation | RMSE | **0.6669** |
|Test (Kaggle) | RMSE | **1.00015** |

---

### 5.2 Interpretation of Results

- The **Validation RMSE of 0.6669** indicates the model predicts monthly shop–item sales with an average error of less than 1 unit (after clipping), which is strong performance given the sparsity and volatility of the data.
- The **Test Validation RMSE of 1.00015** indicates that predictions differ from true monthly sales by about one item per shop–item pair. Given that most monthly quantities fall between 0–3 units, this represents relatively strong forecasting accuracy.


---
### 5.3 Plots

#### RMSE Plot — Actual vs Predicted Sales with RMSE Error Lines

![RMSE Plot](./RMSE_Plot.png)

Plot used mean of observations for better visualization.

---

## 6. Ethical Considerations

This section discusses potential negative impacts, uncertainties, and unexpected results from deploying this forecasting model in a real retail context.

---

### 6.1 Potential Negative Impacts

#### Math / Software Problems
- **Overfitting risk:**  
  If the model becomes overly tailored to historical patterns in 2013–2015 Russian retail data, it may make inaccurate predictions when demand shifts.
- **Feature leakage:**  
  If lag features accidentally use future information, forecasts will appear more accurate than they truly are, leading to misplaced trust.
- **Clipping the target to [0,20]:**  
  Large, rare sales spikes are suppressed — decisions for high-volume items may be wrong.
- **Sparse item/shop combinations:**  
  For items rarely sold, lag features can become misleading, causing extreme underprediction.

#### Real-World Risks
- **Who:** Retail planners, store managers, customers.  
- **What:** Inventory misallocation — too little or too much stock.  
- **When:** During high-demand seasons (holidays), economic shocks, or promotional events not represented in the data.  
- **How:**  
  - **Under-forecasting** → stockouts → lost revenue + customer dissatisfaction  
  - **Over-forecasting** → excess inventory → waste, storage costs, markdown losses  
- These risks intensify when decision makers rely solely on the model without human review.

---

### 6.2 Potential Uncertainties

####  Math / Software Uncertainties
- **Model confidence is not provided:** A low predicted value may be highly uncertain but appear precise.
- **Model is sensitive to hyperparameters:** Different boosting configurations could produce noticeably different outcomes.
- **Unknown causal drivers:**  
  Promotions, competitor pricing, or economic changes are not part of the model, which introduces unknown error.

#### Real-World Uncertainties
- **Who:** Retail companies expanding into new markets.  
- **What:** The model may not generalize to different products or customer behavior.  
- **When:** When sales trends change rapidly — pandemics, supply shocks, trend shifts.  
- **How:**  
  The model assumes the future looks like the past; when that assumption breaks, so does forecasting accuracy.

---

### 6.3 Unexpected or Interesting Results

- **Substantial proportion of zeros:**  
  Many shop–item pairs have no monthly sales, indicating a highly sparse demand distribution — this affects accuracy and makes the binary “any sale?” problem meaningful.
- **Lag features help, but not equally:**  
  Recent lags (1–3 months) influence performance much more than longer seasonal lags (6–12 months).
- **Certain shops/items dominate predictions:**  
  Items with consistent histories are predicted well; niche items remain hard to forecast — highlighting inequality in data representation.

  ---

  ### References

  ChatGPT and Gemini were used to assist in coding and idea generation. 
