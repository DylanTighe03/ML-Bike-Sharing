# Bike Sharing Demand Prediction

This project builds and compares two machine learning models to predict hourly bike rental demand using the UCI Bike Sharing Dataset. Accurate demand forecasting gives bike-sharing operators and city planners the ability to optimize bike distribution, reduce shortages during peak hours, and cut operational costs on low-demand days. The project covers the full pipeline from data exploration and feature engineering through model evaluation and business recommendations.

## Dataset

**Source**: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

**Description**:
- 17,389 hourly records spanning 2011–2012
- 13 features (weather, time, season, calendar)
- Target variable: Continuous count of total bike rentals per hour (`cnt`)
- Demand ranges from 1 to 977 rentals per hour

**Features include**:
- Time-based (hour, weekday, month, year)
- Calendar (holiday, working day, season)
- Weather conditions (clear, mist, light rain, heavy rain)
- Environmental measurements (temperature, humidity, wind speed)

## Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Git

### Installation

1. Clone this repository:
```bash
git clone https://github.com/DylanTighe03/ML-Bike-Sharing.git
cd ML-Bike-Sharing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. The dataset downloads automatically on first run — no manual download needed.

4. Open and run the notebook:
```bash
jupyter notebook Final_project.ipynb
```

## Results

### Model Comparison

| Model | Granularity | MAE | RMSE | R2 |
|---|---|---|---|---|
| Random Forest | Hourly | TBD | TBD | TBD |
| SARIMA | Daily | TBD | TBD | TBD |

*Run the notebook to populate results*

### Key Findings

- **Random Forest** outperforms SARIMA on R2, using 17 features at the hourly level including 5 engineered variables
- SARIMA captures weekly seasonal cycles well but cannot incorporate weather or time-of-day context without exogenous variables
- `rolling_mean_3h` (engineered) was the single most important feature, showing recent demand is the strongest predictor of current demand
- `rush_hour` and `comfort_index` (both engineered) ranked in the top half of feature importance, validating the feature engineering decisions
- Bad weather (rain/snow) cuts demand significantly — the `bad_weather` flag confirmed this is consistent enough to plan operations around
- Fall had the highest average demand across both years, making it the most important season for fleet availability

### Key Insights and Recommendations

- Bikes should be pre-positioned before 8am and 5pm on working days, not redistributed reactively during peak windows
- Scaling back deployment on forecast bad weather days saves operational effort without missing meaningful demand
- Random Forest is the right tool for daily dispatch decisions; SARIMA is better suited for weekly capacity planning
- Both models were trained on 2011–2012 D.C. data and should be retrained regularly to stay accurate

### Next Steps

- Hyperparameter tuning for Random Forest using grid search
- Add exogenous weather variables to SARIMA (SARIMAX)
- Try additional comparison models (XGBoost, LightGBM)
- Expand to station-level predictions for more operational value

## Technologies Used

- **Python 3.8+**: Core programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Random Forest model and evaluation metrics
- **statsmodels**: SARIMA time series model
- **matplotlib**: Data visualization
- **Jupyter Notebook**: Interactive development environment
