import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

def generate_dummy_data(samples=1000):
    np.random.seed(42)
    areas = np.random.randint(500, 5000, samples)
    bedrooms = np.random.randint(1, 6, samples)
    bathrooms = np.random.randint(1, 4, samples)
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], samples)
    
    # Base price calculation
    base_price = 50000
    prices = base_price + (areas * 150) + (bedrooms * 20000) + (bathrooms * 15000)
    
    # Add location multiplier
    location_multipliers = {'Urban': 1.5, 'Suburban': 1.2, 'Rural': 1.0}
    prices = [p * location_multipliers[l] for p, l in zip(prices, locations)]
    
    # Add some noise
    prices = [p + np.random.normal(0, 20000) for p in prices]
    
    df = pd.DataFrame({
        'Area': areas,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Location': locations,
        'Price': prices
    })
    return df

def train_and_save_model():
    print("Generating data...")
    df = generate_dummy_data()
    
    print("Preprocessing data...")
    le = LabelEncoder()
    df['Location_Encoded'] = le.fit_transform(df['Location'])
    
    X = df[['Area', 'Bedrooms', 'Bathrooms', 'Location_Encoded']]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score:.4f}")
    
    print("Saving the model and encoder...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print("Model saved successfully as model.pkl and encoder.pkl")

if __name__ == "__main__":
    train_and_save_model()
