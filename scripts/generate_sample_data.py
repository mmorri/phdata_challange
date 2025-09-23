import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_house_data():
    n_samples = 20000

    data = {
        'id': range(1, n_samples + 1),
        'date': pd.date_range('2014-05-02', periods=n_samples, freq='H').strftime('%Y%m%dT%H%M%S'),
        'price': np.random.lognormal(13.0, 0.5, n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.40, 0.30, 0.08, 0.02]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, p=[0.10, 0.10, 0.30, 0.25, 0.15, 0.05, 0.05]),
        'sqft_living': np.random.randint(370, 13540, n_samples),
        'sqft_lot': np.random.randint(520, 1651359, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.30, 0.10, 0.50, 0.05, 0.05]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'view': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.80, 0.10, 0.05, 0.03, 0.02]),
        'condition': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.01, 0.05, 0.60, 0.30, 0.04]),
        'grade': np.random.choice(range(1, 14), n_samples, p=[0.001, 0.001, 0.002, 0.01, 0.02, 0.08, 0.30, 0.35, 0.15, 0.05, 0.02, 0.01, 0.006]),
        'sqft_above': np.random.randint(370, 9410, n_samples),
        'sqft_basement': np.random.randint(0, 4820, n_samples),
        'yr_built': np.random.randint(1900, 2015, n_samples),
        'yr_renovated': np.random.choice([0] * 19 + list(range(1940, 2015)), n_samples),
        'zipcode': np.random.choice(['98001', '98002', '98003', '98004', '98005', '98006', '98007', '98008',
                                    '98010', '98011', '98014', '98019', '98022', '98023', '98024', '98027',
                                    '98028', '98029', '98030', '98031', '98032', '98033', '98034', '98038',
                                    '98039', '98040', '98042', '98045', '98052', '98053', '98055', '98056',
                                    '98058', '98059', '98065', '98070', '98072', '98074', '98075', '98077',
                                    '98092', '98102', '98103', '98105', '98106', '98107', '98108', '98109',
                                    '98112', '98115', '98116', '98117', '98118', '98119', '98122', '98125',
                                    '98126', '98133', '98136', '98144', '98146', '98148', '98155', '98166',
                                    '98168', '98177', '98178', '98188', '98198', '98199'], n_samples),
        'lat': np.random.uniform(47.1559, 47.7776, n_samples),
        'long': np.random.uniform(-122.519, -121.315, n_samples),
        'sqft_living15': np.random.randint(399, 6210, n_samples),
        'sqft_lot15': np.random.randint(651, 871200, n_samples)
    }

    df = pd.DataFrame(data)

    df['sqft_above'] = df.apply(lambda x: min(x['sqft_above'], x['sqft_living']), axis=1)
    df['sqft_basement'] = df['sqft_living'] - df['sqft_above']
    df['sqft_basement'] = df['sqft_basement'].apply(lambda x: max(0, x))

    df['price'] = df['price'].round(0).astype(int)

    return df

def generate_zipcode_demographics():
    zipcodes = ['98001', '98002', '98003', '98004', '98005', '98006', '98007', '98008',
                '98010', '98011', '98014', '98019', '98022', '98023', '98024', '98027',
                '98028', '98029', '98030', '98031', '98032', '98033', '98034', '98038',
                '98039', '98040', '98042', '98045', '98052', '98053', '98055', '98056',
                '98058', '98059', '98065', '98070', '98072', '98074', '98075', '98077',
                '98092', '98102', '98103', '98105', '98106', '98107', '98108', '98109',
                '98112', '98115', '98116', '98117', '98118', '98119', '98122', '98125',
                '98126', '98133', '98136', '98144', '98146', '98148', '98155', '98166',
                '98168', '98177', '98178', '98188', '98198', '98199']

    n_zipcodes = len(zipcodes)

    data = {
        'zipcode': zipcodes,
        'median_income': np.random.randint(40000, 150000, n_zipcodes),
        'population': np.random.randint(5000, 50000, n_zipcodes),
        'median_age': np.random.uniform(25, 55, n_zipcodes).round(1),
        'total_households': np.random.randint(2000, 20000, n_zipcodes),
        'median_home_value': np.random.randint(200000, 1500000, n_zipcodes),
        'owner_occupied_rate': np.random.uniform(0.3, 0.9, n_zipcodes).round(3),
        'college_degree_rate': np.random.uniform(0.15, 0.75, n_zipcodes).round(3),
        'unemployment_rate': np.random.uniform(0.02, 0.12, n_zipcodes).round(3),
        'crime_index': np.random.uniform(20, 200, n_zipcodes).round(1),
        'school_rating': np.random.randint(3, 10, n_zipcodes)
    }

    return pd.DataFrame(data)

def generate_future_examples():
    n_samples = 100

    data = {
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.40, 0.30, 0.10]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.10, 0.15, 0.35, 0.25, 0.15]),
        'sqft_living': np.random.randint(500, 8000, n_samples),
        'sqft_lot': np.random.randint(1000, 50000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.30, 0.10, 0.50, 0.05, 0.05]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'view': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.75, 0.12, 0.07, 0.04, 0.02]),
        'condition': np.random.choice([2, 3, 4, 5], n_samples, p=[0.05, 0.60, 0.30, 0.05]),
        'grade': np.random.choice(range(4, 12), n_samples, p=[0.01, 0.03, 0.10, 0.35, 0.35, 0.10, 0.04, 0.02]),
        'sqft_above': np.random.randint(500, 6000, n_samples),
        'sqft_basement': np.random.randint(0, 2000, n_samples),
        'yr_built': np.random.randint(1920, 2020, n_samples),
        'yr_renovated': np.random.choice([0] * 90 + list(range(1960, 2020)), n_samples),
        'zipcode': np.random.choice(['98001', '98004', '98006', '98008', '98023', '98027', '98033', '98040',
                                    '98052', '98103', '98105', '98107', '98115', '98117', '98119', '98122',
                                    '98125', '98133', '98144', '98155'], n_samples),
        'lat': np.random.uniform(47.3, 47.7, n_samples),
        'long': np.random.uniform(-122.4, -121.8, n_samples),
        'sqft_living15': np.random.randint(500, 5000, n_samples),
        'sqft_lot15': np.random.randint(1000, 30000, n_samples),
        'price': np.random.lognormal(13.0, 0.5, n_samples).round(0).astype(int)
    }

    df = pd.DataFrame(data)
    df['sqft_above'] = df.apply(lambda x: min(x['sqft_above'], x['sqft_living']), axis=1)
    df['sqft_basement'] = df['sqft_living'] - df['sqft_above']
    df['sqft_basement'] = df['sqft_basement'].apply(lambda x: max(0, x))

    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    print("Generating house data...")
    house_data = generate_house_data()
    house_data.to_csv('data/kc_house_data.csv', index=False)
    print(f"Generated {len(house_data)} house records")

    print("\nGenerating zipcode demographics...")
    demographics = generate_zipcode_demographics()
    demographics.to_csv('data/zipcode_demographics.csv', index=False)
    print(f"Generated demographics for {len(demographics)} zipcodes")

    print("\nGenerating future unseen examples...")
    future_examples = generate_future_examples()
    future_examples.to_csv('data/future_unseen_examples.csv', index=False)
    print(f"Generated {len(future_examples)} test examples")

    print("\nAll data files created successfully!")