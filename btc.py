import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

def load_data(file_path):
    return pd.read_csv(file_path)

def explore_data(data):
    print(data.head())

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Price', data=data)
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.show()

def preprocess_data(data):
    try:
        data['Change %'] = data['Change %'].str.rstrip('%').astype('float') / 100.0

        for col in ['Price', 'Open', 'High', 'Low']:
            data[col] = data[col].replace('[\$,]', '', regex=True).astype(float)

        data['Vol.'] = data['Vol.'].apply(convert_volume)

        features = data[['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']]
        target = (data['Change %'] > 0).astype(int)
        return features, target

    except KeyError as e:
        print(f"KeyError: {e}. En eller flera kolumner saknas i DataFrame.")
        print("Tillgängliga kolumnnamn:")
        print(data.columns)
        raise

def convert_volume(value):
    multiplier = 1

    if 'K' in value:
        multiplier = 1000
    elif 'M' in value:
        multiplier = 1000000
    elif 'B' in value:
        multiplier = 1000000000

    try:
        return float(value.rstrip('KMB')) * multiplier
    except ValueError:
        return 0

def train_model(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')

    return model

def make_prediction(model, latest_data):
    predicted_price_movement = model.predict(latest_data[['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']])

    if predicted_price_movement == 1:
        return "Vi förväntar oss att Bitcoin-priset kommer att öka imorgon."
    else:
        return "Vi förväntar oss att Bitcoin-priset kommer att minska imorgon."

file_path = 'C:\\Users\\jonas\\Desktop\\dataset\\archive\\bitcoin_data.csv'
bitcoin_data = load_data(file_path)

explore_data(bitcoin_data)

features, target = preprocess_data(bitcoin_data)

trained_model = train_model(features, target)

latest_date = bitcoin_data['Date'].max()
latest_price_value = bitcoin_data.loc[bitcoin_data['Date'] == latest_date, 'Price'].values[0]
latest_open_value = bitcoin_data.loc[bitcoin_data['Date'] == latest_date, 'Open'].values[0]
latest_high_value = bitcoin_data.loc[bitcoin_data['Date'] == latest_date, 'High'].values[0]
latest_low_value = bitcoin_data.loc[bitcoin_data['Date'] == latest_date, 'Low'].values[0]
latest_volume_value = bitcoin_data.loc[bitcoin_data['Date'] == latest_date, 'Vol.'].values[0]
latest_change_percentage_value = bitcoin_data.loc[bitcoin_data['Date'] == latest_date, 'Change %'].values[0]

latest_data = pd.DataFrame({
    'Date': [latest_date],
    'Price': [latest_price_value],
    'Open': [latest_open_value],
    'High': [latest_high_value],
    'Low': [latest_low_value],
    'Vol.': [latest_volume_value],
    'Change %': [latest_change_percentage_value]
})

prediction_result = make_prediction(trained_model, latest_data)
print(prediction_result)
