import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    # Rename columns for better readability
    df.rename(columns={
        'YearOfObservation': 'Year_Of_Observation', 
        'Building Dimension': 'Building_Dimension', 
        'NumberOfWindows': 'Number_Of_Windows'
    }, inplace=True)

    # Drop the 'Customer Id' column
    df = df.drop(columns=['Customer Id'])

    # Handle missing values
    mf_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for col in ['Geo_Code', 'Garden']:
        df[col] = mf_imputer.fit_transform(df[[col]]).flatten()  # Flatten the result

    mf_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    df['Building_Dimension'] = mf_imputer.fit_transform(df[['Building_Dimension']]).flatten()  # Flatten the result

    # Encode categorical variables
    ord_categories = [["N", "V"], ["N", "V"], ["O", "V"], ['U', 'R'], ['Fire-resistive', 'Non-combustible', 'Ordinary', 'Wood-framed']]
    enc1 = OrdinalEncoder(categories=ord_categories)
    df[["Building_Painted", "Building_Fenced", "Garden", 'Settlement', 'Building_Type']] = enc1.fit_transform(df[["Building_Painted", "Building_Fenced", "Garden", 'Settlement', 'Building_Type']])

    # Replace categorical values in 'Number_Of_Windows' and convert to integer
    df["Number_Of_Windows"] = df["Number_Of_Windows"].replace({'without': '0', '>=10': '10'}).astype('int')

    # Encode the target variable 'Claim'
    df['Claim'] = df['Claim'].apply(lambda x: 1 if x == 'oui' else 0)

    # Handle outliers in 'Building_Dimension'
    Q1, Q3 = np.percentile(df["Building_Dimension"], [25, 75])
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    df['Building_Dimension'] = np.where(df['Building_Dimension'] >= upper_limit, upper_limit, 
                                         np.where(df['Building_Dimension'] <= lower_limit, lower_limit, df['Building_Dimension']))

    # Label encoding for 'Geo_Code'
    le = LabelEncoder()
    df['Geo_Code'] = le.fit_transform(df['Geo_Code'])

    return df
   

def handle_class_imbalance(df):
    # Upsample the minority class
    df_majority = df[df['Claim'] == 0]
    df_minority = df[df['Claim'] == 1]

    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    return pd.concat([df_majority, df_minority_upsampled])


def stacking_model(X, Y):
    base_models = [
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(random_state=42)),
        ('mlp', MLPClassifier(random_state=42, max_iter=300))
    ]
    meta_model = GradientBoostingClassifier(random_state=42)
    stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_test)

    print("Stacking Model Accuracy:", accuracy_score(y_test, y_pred_stack))
    print(classification_report(y_test, y_pred_stack))


# Main execution
def main():
    filepath = "test_Insurance.csv"
    insurance_data = load_data(filepath)
    processed_data = preprocess_data(insurance_data)
    
    # Handle class imbalance
    balanced_data = handle_class_imbalance(processed_data)
    
    # Split features and target
    X = balanced_data.drop('Claim', axis=1)
    Y = balanced_data['Claim']

    # Stacking model
    stacking_model(X,Y)

if __name__ == "__main__":
    main()