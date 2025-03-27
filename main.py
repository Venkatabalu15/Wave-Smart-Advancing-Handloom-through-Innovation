import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def train_inventory_model():
    data = pd.read_csv("data/demand.csv")
    X = data.drop("inventory_total", axis=1)
    y = data["inventory_total"]
    
    numeric_features = ["price", "units", "uses_ad", "product_inventory"]
    categorical_features = ["color", "size", "countries"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print_model_metrics(y_test, y_pred)
    
    return model, preprocessor

def train_price_model():
    data = pd.read_csv("data/pricing.csv")
    X = data.drop("Price", axis=1)
    y = data["Price"]
    
    categorical_features = ["Design", "Category", "Color", "Size", "Material"]
    numerical_features = ["Year"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print_model_metrics(y_test, y_pred)
    
    return model, preprocessor

def print_model_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=== Model Performance ===")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}\n")

def predict_inventory(model, preprocessor, price, units, uses_ad, color, size, product_inventory, countries):
    new_data = pd.DataFrame({
        "price": [price],
        "units": [units],
        "uses_ad": [uses_ad],
        "color": [color],
        "size": [size],
        "product_inventory": [product_inventory],
        "countries": [countries]
    })
    
    new_data_processed = preprocessor.transform(new_data)
    prediction = model.predict(new_data_processed)[0]
    return max(0, round(prediction, 2))

def predict_price(model, preprocessor, Design, Category, Color, Size, Material, Year=2025):
    new_product = pd.DataFrame({
        "Design": [Design],
        "Category": [Category],
        "Color": [Color],
        "Size": [Size],
        "Material": [Material],
        "Year": [Year]
    })
    
    new_product_processed = preprocessor.transform(new_product)
    predicted_price = model.predict(new_product_processed)[0]
    return max(0, round(predicted_price, 2))

def plot_price_trend(model, preprocessor, Category, Color, Size, Material, year=2025):
    designs = ["Eri", "Narayanpet", "Gingham"]
    predicted_prices = []
    for design in designs:
        predicted_price = predict_price(model, preprocessor, design, Category, Color, Size, Material, year)
        predicted_prices.append(predicted_price)
    plt.figure(figsize=(8, 5))
    plt.plot(designs, predicted_prices, marker='o', linestyle='-', color='b', label='Predicted Price')
    plt.xlabel("Design pattern")
    plt.ylabel("Price (rupees/dress)")
    plt.title("Predicted Price Trend for 3 different Design patterns")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    inventory_model, inventory_preprocessor = train_inventory_model()
    price_model, price_preprocessor = train_price_model()
    
    inventory_prediction = predict_inventory(inventory_model, inventory_preprocessor, 100, 5, 1, "red", "M", 50, 15)
    print(f"Predicted Inventory: {inventory_prediction} tons")
    
    price_prediction = predict_price(price_model, price_preprocessor, "Kantha", "Shirt", "Blue", "L", "Cotton")
    print(f"Predicted Price: {price_prediction} rupees/dress")
    
    plot_price_trend(price_model, price_preprocessor, "Shirt", "Blue", "L", "Cotton")
