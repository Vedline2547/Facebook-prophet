# Time Series Forecasting with Facebook Prophet

## Project Overview

This project explores time series forecasting using **Facebook Prophet**, a forecasting library designed to model time-dependent data and capture trends and seasonal patterns.

Using the AirPassengers dataset, the project analyzes historical monthly airline passenger numbers and forecasts passenger traffic for the next 12 months. A logarithmic transformation is applied to the passenger counts before training the model to help manage the increasing variation in the series.

The project also demonstrates how to transform predictions back to the original scale, visualize forecast components, and compare actual passenger counts with predicted values using Matplotlib and Plotly.

## Objectives

- Explore and prepare time series data for forecasting.
- Apply logarithmic transformation to passenger counts.
- Train a Prophet forecasting model.
- Generate forecasts for the next 12 months.
- Analyze trend and seasonal components.
- Transform forecast values and uncertainty intervals back to the original scale.
- Visualize historical data, predictions, and uncertainty intervals.

## Dataset

**AirPassengers Dataset**

The dataset contains monthly airline passenger totals from 1949 to 1960.

| Feature | Description |
|---|---|
| `Month` | Month and year of observation |
| `#Passengers` | Number of airline passengers |

Before training the model, the columns are renamed to `ds` and `y`, as required by Prophet.

- `ds`: Date column.
- `y`: Target variable representing passenger counts.

The target variable is transformed using the natural logarithm:

```python
df['y'] = np.log(df['y'])
```

This transformation helps manage the increasing variance in the original passenger series.

## Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical computations and logarithmic transformations
- **Facebook Prophet** — Time series forecasting
- **Matplotlib** — Forecast visualization
- **Plotly** — Interactive visualization

## Project Workflow

### 1. Data Preparation
Load the AirPassengers dataset, rename the columns to Prophet's required format, convert the date column to datetime, and preserve the original passenger counts.

### 2. Logarithmic Transformation
Apply a natural logarithm to the passenger counts before model training.

### 3. Model Training
Initialize and fit a Prophet model using the historical monthly observations.

### 4. Future Forecasting
Generate a future dataframe covering 12 additional months and use the trained model to produce forecasts.

### 5. Forecast Visualization
Use Prophet's plotting functions to visualize the forecast and its components, including trend and applicable seasonal patterns.

### 6. Inverse Transformation
Apply the exponential function to the predicted values and uncertainty bounds to return the forecasts to the original passenger-count scale.

### 7. Interactive Visualization
Use Plotly to compare actual historical passenger counts with predicted values and their uncertainty intervals.

## Forecasting Approach

The project follows these main steps:

1. Load and preprocess the historical data.
2. Transform passenger counts using the natural logarithm.
3. Fit the Prophet model.
4. Generate a 12-month forecast.
5. Convert predictions back to the original scale.
6. Visualize actual and predicted passenger counts.

## How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install prophet pandas numpy matplotlib plotly
```

### Clone the Repository

```bash
git clone https://github.com/Vedline2547/Facebook-prophet.git
```

Navigate to your project directory, or replace the repository URL with the URL of this forecasting repository if it is stored separately.

```bash
cd Facebook-prophet
```

### Run the Script

```bash
python main.py
```

Ensure that `AirPassengers.csv` is available at the path specified in the script, or update the file path to match your local setup.

## Expected Outputs

- Forecast visualization showing historical observations and predicted passenger counts.
- Trend and seasonal component plots.
- Forecasts for 12 future months.
- Upper and lower uncertainty bounds.
- Interactive Plotly visualization comparing actual and predicted passenger traffic.

## Key Learnings

Through this project, I explored:

- Time series forecasting with Facebook Prophet.
- Preparing date-indexed data for forecasting.
- Logarithmic and inverse transformations.
- Trend and seasonality analysis.
- Forecast uncertainty visualization.
- Creating interactive data visualizations with Plotly.

## Future Improvements

- Evaluate forecast accuracy using a chronological train-test split.
- Calculate MAE, RMSE, and other appropriate forecasting metrics.
- Compare Prophet with ARIMA and other forecasting approaches.
- Experiment with changepoints and seasonal parameters.
- Improve forecast evaluation using a rolling or expanding time-series validation strategy.

## Conclusion

This project demonstrates a practical time series forecasting workflow using Facebook Prophet. By combining data preprocessing, logarithmic transformation, forecasting, inverse transformation, and visualization, it provides a foundation for exploring time-dependent patterns and predicting future passenger traffic.

## Author

**Vedline Ochieng**

- GitHub: [Vedline2547](https://github.com/Vedline2547)
- LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

*This project is part of my ongoing journey in Python, machine learning, and data science.*
