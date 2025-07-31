import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocessing(csv_path):
    df = pd.read_csv(csv_path)

    # Encode categorical variables
    cat_features = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
    le = LabelEncoder()
    for col in cat_features:
        df[col] = le.fit_transform(df[col])

    df['Has_Hypertension'] = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['Has_Hypertension'])
    y = df['Has_Hypertension']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    trainX, testX, trainY, testY = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return trainX, testX, trainY, testY, scaler
