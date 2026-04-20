#  Chicago Crime Intelligence Dashboard

A robust, interactive **Streamlit** dashboard and data pipeline built to analyze, visualize, and forecast crime patterns across Chicago. This application provides high-level metrics, geospatial heatmaps, and time-series forecasting using an elegant, fully responsive dark-mode UI.

##  Features

- **Interactive UI**: A highly polished, custom-styled dark theme dashboard utilizing modern typography (Outfit Font).
- **Data Filtering**: Filter the entire dataset instantly by Year, Crime Type, and District via the intelligence sidebar.
- **Geospatial Mapping**: Visualize crime hotspots across Chicago through interactive geographic heatmaps using Folium and CartoDB Dark Matter tiles.
- **Deep Data Visualizations**: Rich, colorful charts covering:
  - Yearly & Monthly Crime Trends
  - Top 10 Crime Types
  - Crime Distribution by Hour of the Day
  - KPI Metrics (Total Crimes, Most Common Crimes, Arrest Rates, etc.)
- **Forecasting Module**: Built-in Linear Regression model (`scikit-learn`) for predicting future monthly crime counts based on historical data.

##  Technology Stack

- **Frontend/UI**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Geospatial & Mapping**: Folium
- **Machine Learning**: Scikit-Learn

## 📂 Project Structure

```text
├── app/
│   └── app.py               # Main Streamlit dashboard application
├── data/                    # Directory containing cleaned dataset (cleaned_chicago_crime.csv)
├── models/                  # Directory storing trained ML model binaries (.pkl)
├── .streamlit/              # Custom Streamlit UI configuration (forces Dark Mode)
├── eda.py                   # Exploratory Data Analysis & data aggregation module
├── forecasting.py           # ML module for training the Linear Regression model
└── requirements.txt         # Required Python dependencies
```

##  Getting Started

Follow these steps to set up the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/May-nk/chicago-crime-dashboard.git
cd chicago-crime-dashboard
```

### 2. Set up a virtual environment (Recommended)

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required packages. *(Note: Ensure `scikit-learn` is installed if you intend to run the forecasting models).*

```bash
pip install -r requirements.txt
pip install scikit-learn
```

### 4. Data Preparation

The dashboard relies on local data to operate. Please ensure that the cleaned CSV dataset is located exactly at:
`data/cleaned_chicago_crime.csv`

### 5. Run the Application

Execute the following command to start the Streamlit server:

```bash
streamlit run app/app.py
```

The application will automatically pop up in your default web browser at `http://localhost:8501`.

## 🤖 Running the Forecasting Model

You can run the forecasting script independently to train the time-series forecasting model without spinning up the dashboard.

```bash
python forecasting.py
```

This script will:
1. Aggregate historical monthly data.
2. Train a scikit-learn `LinearRegression` model.
3. Print out accuracy metrics (MAE and RMSE).
4. Save the compiled model into `models/forecast_model.pkl`.
