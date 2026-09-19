# Predictive House Price Modeling

A reproducible scikit-learn regression project using a synthetic housing dataset. It demonstrates feature engineering, categorical one-hot encoding, scaling, and RMSE/R-squared evaluation.

## Run
```bash
python -m pip install -r requirements.txt
python src/train.py
```
The saved pipeline and metrics are written to `artifacts/`.

## Predict a home
```sh
python src/predict.py --area 1500 --bedrooms 3 --age 10 --neighborhood Central
```
The synthetic price formula uses floor area, bedrooms, building age, neighborhood and random noise. It is for demonstrating machine-learning workflow, not estimating real property values. The 80/20 train/test split is reproducible; the encoder and scaler are fitted only on training data. The test result is RMSE approximately $29,603 and R-squared 0.916. Training creates the model locally; serialized models are excluded from Git to avoid version incompatibilities.
