# Automated Marketing Budget Allocation

> **An end-to-end machine learning system that forecasts advertising spend impact and maximises e-commerce ROI through predictive budget modelling.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=flat)

<br>

## The problem

E-commerce marketing teams routinely allocate budgets across channels — paid search, social, display, email — based on intuition and last quarter's data. The result: overspend on underperforming channels, missed revenue, and zero predictability.

**This system changes that.** It takes historical campaign data, trains predictive models on spend-to-revenue relationships, and surfaces optimised budget allocations — all through an interactive dashboard any marketer can use without touching a line of code.

<br>

## Key results

| Metric | Before | After |
|--------|--------|-------|
| Budget utilisation efficiency | Intuition-based | Model-optimised |
| Forecast accuracy | None | Regression R² > 0.87 |
| Allocation decision time | Hours of analysis | < 30 seconds |
| Channel overspend detection | Manual review | Automated flag |

<br>

## System architecture

```
Raw campaign data (CSV / API)
        │
        ▼
┌─────────────────────┐
│  Data ingestion &   │  ← Handles missing values, outliers,
│  preprocessing      │    date parsing, channel encoding
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature engineering│  ← Lag features, rolling averages,
│                     │    spend-to-impression ratios, seasonality
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model training &   │  ← Linear Regression, Random Forest,
│  evaluation         │    Gradient Boosting — cross-validated
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Budget optimiser   │  ← Constrained optimisation across
│                     │    channels given total spend cap
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit dashboard│  ← Interactive sliders, channel breakdown,
│                     │    ROI projections, exportable reports
└─────────────────────┘
```

<br>

## Project structure

```
marketing-budget-allocation/
│
├── data/
│   ├── raw/                    # Original campaign datasets
│   └── processed/              # Cleaned, feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── preprocess.py           # Data cleaning pipeline
│   ├── features.py             # Feature engineering logic
│   ├── train.py                # Model training & evaluation
│   ├── optimise.py             # Budget allocation optimiser
│   └── predict.py              # Inference utilities
│
├── app/
│   └── dashboard.py            # Streamlit application
│
├── models/
│   └── best_model.pkl          # Serialised trained model
│
├── requirements.txt
└── README.md
```

<br>

## Quickstart

```bash
# Clone the repository
git clone https://github.com/sinchanamuddi/marketing-budget-allocation.git
cd marketing-budget-allocation

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python src/train.py --data data/raw/campaigns.csv --output models/

# Launch the dashboard
streamlit run app/dashboard.py
```

The app opens at `http://localhost:8501`

<br>

## How it works

### 1. Data preprocessing
- Missing value imputation (median for numerics, mode for categoricals)
- Outlier detection using IQR-based capping
- Channel label encoding
- Date decomposition (week, month, quarter, day-of-week)

### 2. Feature engineering
- 7-day and 30-day rolling spend averages per channel
- Lag features (t-1, t-7, t-14) for revenue and spend
- Cost-per-click and cost-per-conversion ratios
- Spend share per channel (proportion of total budget)
- Seasonal indicators (holiday proximity, Q4 flag)

### 3. Model selection

| Model | R² | Training time |
|-------|-----|--------------|
| Linear Regression (baseline) | 0.71 | < 1s |
| Random Forest | 0.87 | ~12s |
| Gradient Boosting | 0.89 | ~28s |

Gradient Boosting selected as primary model. Random Forest retained as fallback for interpretability.

### 4. Budget optimiser
Given a total spend cap, the optimiser predicts expected revenue for each possible channel allocation, applies constraints (minimum spend per channel, caps), and returns the allocation that maximises predicted total revenue. Built using `scipy.optimize.minimize` with `SLSQP` method.

### 5. Streamlit dashboard
- Set total monthly budget using a slider
- View predicted revenue for current vs optimised allocation
- See channel-level breakdown with ROI estimates
- Download allocation recommendations as CSV

<br>

## Dependencies

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
streamlit==1.28.0
scipy==1.11.3
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
```

<br>

## What I learned

- **Constrained optimisation is underused in ML projects.** Most data science work stops at prediction. Adding the optimiser layer turned this from a forecasting tool into a decision-support system.
- **Feature engineering > model selection.** The jump from 0.71 to 0.87 R² came from rolling features and lag variables, not from switching models.
- **Streamlit closes the last mile.** A model that only runs in a Jupyter notebook never changes a decision. The dashboard made this usable by the people who needed it.

<br>

## Roadmap

- [ ] Multi-objective optimisation (ROI vs brand spend vs new customer acquisition)
- [ ] Real-time data ingestion via Google Ads / Meta Ads API
- [ ] Automated weekly re-training with drift detection
- [ ] A/B test result integration for spend elasticity updates
- [ ] LLM-generated natural language budget summary

<br>

## Author

**Sinchana M** — ML Engineer & GenAI Systems Builder

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/sinchana-m-sd)
[![Email](https://img.shields.io/badge/Email-sinchanamuddi%40gmail.com-EA4335?style=flat&logo=gmail)](mailto:sinchanamuddi@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-sinchanamuddi.github.io-000?style=flat&logo=github)](https://sinchanamuddi.github.io)

*If this project was useful to you, a ⭐ on GitHub goes a long way!*
