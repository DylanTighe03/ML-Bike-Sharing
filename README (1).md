# Bike Sharing Demand Prediction

This project predicts hourly bike rental demand using machine learning and time series models on the UCI Bike Sharing Dataset. By accurately forecasting demand, bike-sharing operators and city planners can optimize bike distribution and reduce shortages during peak hours.

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
git clone https://github.com/yourusername/bike-sharing-demand.git
cd bike-sharing-demand
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. The dataset downloads automatically on first run — no manual download needed.

4. Open and run the notebook:
```bash
jupyter notebook final_project_draft.ipynb
```

### Key Findings

- **Random Forest** outperforms SARIMA on R2, using 14 features at the hourly level
- SARIMA captures weekly seasonal cycles well but cannot incorporate weather or time-of-day context without exogenous variables
- `rolling_mean_3h` (engineered) was the single most important feature, showing recent demand is the strongest predictor of current demand
- `rush_hour` and `comfort_index` (both engineered) ranked in the top half of feature importance, validating the feature engineering decisions

### Next Steps

- Hyperparameter tuning for Random Forest using grid search
- Add exogenous weather variables to SARIMA (SARIMAX)
- Try tradditional comparison models
- Expand to station level predictions for more operational value

## Technologies Used

- **Python 3.8+**: Core programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Random Forest model and evaluation metrics
- **statsmodels**: SARIMA time series model
- **matplotlib**: Data visualization
- **Jupyter Notebook**: Interactive development environment
