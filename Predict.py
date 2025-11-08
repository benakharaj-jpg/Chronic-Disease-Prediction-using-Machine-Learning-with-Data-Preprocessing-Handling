import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def process(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14):
    # Load dataset
    data = pd.read_csv("./Dataset/demotest.csv")
    
    # Extract features and labels
    X = data.iloc[:, 0:14]  # Features
    y = data.iloc[:, 14]  # Labels

    # Convert categorical target variable to numeric if needed
    y = y.astype(int)

    # Train-test split for validation
    X_train, X_test_split, y_train, y_test_split = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling for better performance
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test_split)

    # Train logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0)
    model.fit(X_train, y_train)

    # Model accuracy check
    y_test_pred = model.predict(X_test_split)
    test_accuracy = accuracy_score(y_test_split, y_test_pred) * 100  # Convert to percentage

    # Preparing input data for prediction
    input_data = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14]).reshape(1, -1)
    #input_data_scaled = scaler.transform(input_data)
    #print("input_data_scaled==",input_data_scaled)

    # Make prediction
    
    X_test =input_data
    print("Testing data",X_test)
    y_pred = model.predict(X_test)

    # Mapping predictions to diseases
    disease_mapping = {
        1: ("Heart Disease", "Lifestyle changes such as eating a diet low in salt and saturated fat, getting more exercise, and not smoking. Medicines. A heart procedure. Heart surgery."),
        2: ("Hypertension", "Water pills (diuretics), ACE inhibitors, ARBs, Calcium channel blockers."),
        3: ("Stress", "Incorporating relaxation techniques like meditation or yoga, engaging in regular physical activity, maintaining a healthy diet, and seeking social support."),
        4: ("Kidney Disease", "ACE inhibitors, ARBs, diuretics, phosphate binders, cholesterol-lowering drugs, erythropoietin, Vitamin D."),
        5: ("No Disease", "Continue your regular exercise.")
    }

    result, treatment = disease_mapping.get(y_pred[0], ("Unknown", "Consult a doctor for further analysis."))

    return result, treatment, test_accuracy  # Returning accuracy along with result & treatment
##process("finaldataset.csv",0,46,1,23,0,0,0,1,285,130,84,23.1,85,130)# Heart Valve Disease
##process("finaldataset.csv",0,61,1,30,0,0,1,0,225,150,95,28.58,65,103)# Coronary Artery Disease
##process("finaldataset.csv",1,48,1,20,0,0,0,0,245,127.5,80,25.34,75,70) #Heart Arrhythmias
##process("dataset.csv",1,43,1,30,0,0,1,0,225,162,107,23.61,93,88)# Pericardial Disease


