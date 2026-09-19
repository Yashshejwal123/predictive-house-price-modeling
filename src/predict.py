"""Predict a single home with the locally trained pipeline."""
import argparse
from pathlib import Path
import joblib
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--area', type=float, required=True, help='Floor area in square feet')
parser.add_argument('--bedrooms', type=int, required=True)
parser.add_argument('--age', type=int, required=True, help='Age in years')
parser.add_argument('--neighborhood', choices=['Central', 'North', 'East', 'West'], required=True)
args = parser.parse_args()
if args.area <= 0 or args.bedrooms < 1 or args.age < 0:
    parser.error('Area and bedrooms must be positive; age cannot be negative')
path = Path(__file__).resolve().parents[1] / 'artifacts/house_price_pipeline.joblib'
if not path.exists():
    parser.error('Run python src/train.py first')
model = joblib.load(path)
home = pd.DataFrame([dict(area_sqft=args.area, bedrooms=args.bedrooms, age_years=args.age, neighborhood=args.neighborhood)])
print(f'Illustrative predicted price (USD): ${model.predict(home)[0]:,.0f}')
