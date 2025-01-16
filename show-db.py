# source venv/bin/activate
import tensorflow as tf
import pandas as pd
from ucimlrepo import fetch_ucirepo

automobile = fetch_ucirepo(id=10)
X = automobile.data.features  

print("Atrybuty dla kazdego samochodu:")
for index, row in X.iterrows():
    print(f"Samochód {index + 1}:")
    for column in X.columns:
        print(f"  {column}: {row[column]}")
    print("-" * 30)