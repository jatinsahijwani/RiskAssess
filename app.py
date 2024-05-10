import joblib
from flask import Flask, request,jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/predictkidney", methods=['POST'])
def predictkidney():
    if request.method == "POST":
        data = request.get_json()
        to_predict_list = list(data.values()) 
        to_predict_list = list(map(float, to_predict_list))  
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
            if int(result) == 1:
                prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
            else:
                prediction = "Patient has a low risk of Kidney Disease"
            return jsonify({'prediction': prediction})
    return jsonify({'error': 'Invalid request'}), 400 


def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predictliver', methods=['POST'])
def predictliver():
    if request.method == "POST":
        data = request.get_json() 
        to_predict_list = list(data.values())  
        to_predict_list = list(map(float, to_predict_list))  
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)
            if int(result) == 1:
                prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
            else:
                prediction = "Patient has a low risk of Liver Disease"
            return jsonify({'prediction': prediction}) 
    return jsonify({'error': 'Invalid request'}), 400  


##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


if __name__ == "__main__":
    app.run(debug=True)

