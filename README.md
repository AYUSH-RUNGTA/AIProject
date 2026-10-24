# AIProject

# Food Delivery Time Predictor

This project predicts how many minutes a food delivery will take. It uses about 45k real Zomato orders from Kaggle and compares two models:

- **Random Forest**, a classic ML model
- **Neural Network**, scikit-learn's `MLPRegressor`

## Run
```
streamlit run app.py    # website (trains the models when it starts)
python main.py          # optional: run the pipeline on its own and see the plots
```

## Files
- `main.py`: the full pipeline: load, clean, explore, train and compare. Nothing is saved to disk.
- `app.py`: a website that imports `main.py`. It shows which model is better, 4 plots, and the predicted time for an order you enter.
