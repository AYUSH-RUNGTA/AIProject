# AIProject


# 🛵 Food Delivery Time Predictor

Predict how many minutes a food delivery will take. The project uses about 45,000 real Zomato deliveries and compares a **Random Forest** with a **Neural Network**.

The project covers the full data science pipeline: loading, cleaning, exploring, training, testing and comparing. A simple Streamlit website lets anyone try the models.

## Dataset

[Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset) on Kaggle: 45,593 deliveries from Indian cities, with 20 columns.

Each row is one delivery. It records the rider (age, rating), the conditions (weather, traffic, festival day, city type), the order (food type, vehicle, vehicle condition, extra deliveries on the same trip) and the **time taken in minutes**, which is the target.

The raw data is messy, so cleaning is a real step:
- delivery times are stored as text like `"(min) 24"`
- missing values are stored as the text `"NaN"` instead of real blanks
- weather is written as `"conditions Sunny"`, with extra spaces in many columns
- numbers such as age and rating are stored as text

## Pipeline (`main.py`)

1. **Load:** download the dataset with `kagglehub`.
2. **Clean:** fix the text values, drop missing rows and convert columns to the right types.
3. **Explore:** average delivery times by traffic and weather, plus 4 plots.
4. **Prepare:** one-hot encode the text columns (`pd.get_dummies`), make an 80/20 train/test split and scale the features (`StandardScaler`).
5. **Train:** `RandomForestRegressor` and `MLPRegressor`, a neural network with layers of 64 and 32.
6. **Compare:** score both models on the test set with MAE and R², and pick the winner.

## Results

| Model | Average error (MAE) | R² |
|---|---|---|
| Random Forest | 4.09 min | 0.675 |
| **Neural Network** | **3.94 min** | **0.710** |

The Neural Network wins: on deliveries it has never seen, its guesses are off by about 4 minutes on average.

What the data shows:
- The average delivery takes **26.6 minutes**.
- Traffic matters most: about **21 min** in low traffic and **31 min** in a jam.
- Weather matters too: about **22 min** when sunny and **29 min** in fog or cloudy weather.

## Website (`app.py`)

The Streamlit app trains the models when it starts, then shows:
- the model comparison and the winner
- a dropdown to view each plot
- a form where you enter an order and get a prediction from both models

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first start takes about 30 seconds while the models train. You can also run `python main.py` to see the results in the terminal and the plots in windows.

> If `kagglehub` asks you to log in, create an API token on Kaggle (Settings, then API) and put `kaggle.json` in `~/.kaggle/`.

## Project structure

```
delivery-time-predictor/
├── main.py           # data pipeline: load, clean, explore, train, compare
├── app.py            # Streamlit website
├── requirements.txt
└── README.md
```

## Tech used

Python, pandas, scikit-learn, matplotlib, Streamlit, kagglehub

## Possible improvements

- Add the **delivery distance**, calculated from the restaurant and customer GPS coordinates. In an earlier version this cut the error to about 3.2 minutes.
- Use the order and pickup times to measure how long the food took to prepare.
- Tune the model settings (hyperparameters).
