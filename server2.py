import os
import json
import joblib
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import SqliteDatabase, Model, IntegerField, FloatField, TextField, IntegrityError,CharField
from sklearn.base import BaseEstimator, TransformerMixin
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect



# class TotalChargesCleaner(BaseEstimator, TransformerMixin):
#     """Removes currency symbols and converts Total Charges to float."""
#     def __init__(self, column="Total Charges"):
#         self.column = column

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         X[self.column] = X[self.column].replace(r'[\$,]', '', regex=True).astype(float)
#         return X
    
class APRRiskMapper(BaseEstimator, TransformerMixin):
    """Maps APR Risk of Mortality from categorical to ordinal numbers."""
    def __init__(self, column="APR Risk of Mortality", mapping=None):
        self.column = column
        self.mapping = mapping or {"Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].map(self.mapping)
        return X

class ConvertToStringTransformer(BaseEstimator, TransformerMixin):
    """Converts specified columns to string type."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(str)
        return X     

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = CharField(primary_key=True, max_length=50)
    observation = TextField()
    true_value = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

with open('column_names.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    obs_dict = request.get_json()

    observation_id = obs_dict.get("observation_id", None)
    data = obs_dict.get("data", None)

    if observation_id is None:
        return jsonify({"error": "Missing 'observation_id'", "observation_id": None}), 200  

    if data is None:
        return jsonify({"error": "Missing 'data'", "observation_id": observation_id}), 200  
    

    #check this

    #extra_columns = [key for key in data.keys() if key not in columns]
    
    #if extra_columns:
        #return jsonify({
            #"error": f"Unexpected column(s): {', '.join(extra_columns)}",
            #"observation_id": observation_id
        #}), 200# 

    missing_columns = [col for col in columns if col not in data]
    if missing_columns:
       return jsonify({
            "error": f"Missing column(s): {', '.join(missing_columns)}",
            "observation_id": observation_id
        }), 200  
    

    valid_race = ["White", "Other Race", "Black/African American", "Multi-racial"]
    valid_gender = ["M", "F"]
    valid_ethnicity= ["Not Span/Hispanic", "Unknown", "Spanish/Hispanic", "Multi-ethnic"]

    valid_agegroup=["0 to 17", "18 to 29", "70 or Older", "50 to 69", "30 to 49"]

    valid_type_of_admission=["Emergency", "Elective", "Urgent", "Newborn", "Not Available","Trauma"]


    # valid_patient_disposition=['Home or Self Care', 'Another Type Not Listed',
    #    'Home w/ Home Health Services', 'Short-term Hospital',
    #    'Skilled Nursing Home', 'Left Against Medical Advice',
    #    'Hospice - Home', 'Hosp Basd Medicare Approved Swing Bed',
    #    'Expired', 'Facility w/ Custodial/Supportive Care',
    #    'Psychiatric Hospital or Unit of Hosp',
    #    'Inpatient Rehabilitation Facility',
    #    'Federal Health Care Facility', 'Court/Law Enforcement',
    #    'Hospice - Medical Facility',
    #    "Cancer Center or Children's Hospital",
    #    'Medicare Cert Long Term Care Hospital',
    #    'Critical Access Hospital', 'Medicaid Cert Nursing Facility']
    


    valid_apr_risk_of_mortality=['Minor', 'Moderate', 'Major', 'Extreme']

    valid_abortion_indicator=['N', 'Y']

    if data["Abortion Edit Indicator"] not in valid_abortion_indicator:
        return jsonify({
        "error": f"Invalid value for 'Abortion Edit Indicator': {data['Abortion Edit Indicator']}. Valid values are: {', '.join(valid_abortion_indicator)}",
        "observation_id": observation_id
        }), 200 


    if data["APR Risk of Mortality"] not in valid_apr_risk_of_mortality:
        return jsonify({
        "error": f"Invalid value for 'APR Risk of Mortality': {data['APR Risk of Mortality']}. Valid values are: {', '.join(valid_apr_risk_of_mortality)}",
        "observation_id": observation_id
        }), 200 
    

    # if data["Patient Disposition"] not in valid_patient_disposition:
    #     return jsonify({
    #     "error": f"Invalid value for 'Patient Disposition': {data['Patient Disposition']}. Valid values are: {', '.join(valid_patient_disposition)}",
    #     "observation_id": observation_id
    #     }), 200 


    if data["Type of Admission"] not in valid_type_of_admission:
        return jsonify({
        "error": f"Invalid value for 'Type of Admission': {data['Type of Admission']}. Valid values are: {', '.join(valid_type_of_admission)}",
        "observation_id": observation_id
        }), 200 


    if data["Age Group"] not in valid_agegroup:
        return jsonify({
            "error": f"Invalid value for 'Age Group': {data['Age Group']}. Valid values are: {', '.join(valid_agegroup)}",
            "observation_id": observation_id
        }), 200  


    if data["Ethnicity"] not in valid_ethnicity:
        return jsonify({
           "error": f"Invalid value for 'Ethnicity': {data['Ethnicity']}. Valid values are: {', '.join(valid_ethnicity)}",
            "observation_id": observation_id
        }), 200  

    if data["Gender"] not in valid_gender:
        return jsonify({
            "error": f"Invalid value for 'Gender': {data['Gender']}. Valid values are: {', '.join(valid_gender)}",
            "observation_id": observation_id
        }), 200  

    if data["Race"] not in valid_race:
        return jsonify({
            "error": f"Invalid value for 'Race': {data['Race']}. Valid values are: {', '.join(valid_race)}",
            "observation_id": observation_id
        }), 200 
    
    
    # Validar valores numéricos


    #data["Total Charges"]=data["Total Charges"].str.replace("$", "", regex=False).astype(float)


    try:
        Facility_Id = float(data["Facility Id"])
    except (ValueError, TypeError):
        return jsonify({
            "observation_id": observation_id,
            "error": f"Facility Id: {data['Facility Id']} is not a valid number"
        }), 200

    try:
        CCS_Diagnosis_Code = int(data["CCS Diagnosis Code"])
        if CCS_Diagnosis_Code < 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({
            "observation_id": observation_id,
            "error": f"CCS Diagnosis Code: {data['CCS Diagnosis Code']} must be a positive integer"
        }), 200

    try:
        CCS_procedure_Code = int(data["CCS Procedure Code"])
        if CCS_procedure_Code < 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({
            "observation_id": observation_id,
            "error": f"CCS Procedure Code: {data['CCS Procedure Code']} must be a positive integer"
        }), 200

    try:
        APR_DRG_Code = int(data["APR DRG Code"])
        if APR_DRG_Code < 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({
            "observation_id": observation_id,
            "error": f"APR DRG Code: {data['APR DRG Code']} must be a positive integer"
        }), 200

    try:
        APR_MDC_Code = int(data["APR MDC Code"])
        if APR_MDC_Code < 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({
            "observation_id": observation_id,
            "error": f"APR MDC Code: {data['APR MDC Code']} must be a positive integer"
        }), 200

    try:
        APR_Severity_of_Illness_Code = int(data["APR Severity of Illness Code"])
        if APR_Severity_of_Illness_Code < 0 or APR_Severity_of_Illness_Code > 4:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({
            "observation_id": observation_id,
            "error": f"APR Severity of Illness Code: {data['APR Severity of Illness Code']} must be an integer between 0 and 4"
        }), 200

    try:
        Birth_Weight = int(data["Birth Weight"])
        if Birth_Weight < 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({
            "observation_id": observation_id,
            "error": f"Birth Weight: {data['Birth Weight']} must be a positive integer"
        }), 200



    input_data = {col: data.get(col) for col in columns}
    input_df = pd.DataFrame([input_data]).astype(dtypes)

    try:
        # Generate prediction first
        prediction = int(pipeline.predict(input_df)[0])  # <-- PREDICTION HAPPENS HERE

        # Then check if record exists and update/create
        try:
            p = Prediction.get(Prediction.observation_id == observation_id)
            # Update existing record
            p.observation = json.dumps(data)
            p.prediction = prediction  # <-- Store prediction (not true_value!)
            p.save()
        except Prediction.DoesNotExist:
            # Create new record
            p = Prediction.create(
                observation_id=observation_id,
                observation=json.dumps(data),
                prediction=prediction  # <-- Store prediction (not true_value!)
            )

        return jsonify({
            "observation_id": observation_id,
            "prediction": prediction
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
   

@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_value = obs['true_value']
        p.save()
        return jsonify(model_to_dict(p)), 200
    except Prediction.DoesNotExist:
        error_msg = f'Observation ID: "{obs["observation_id"]}" does not exist'
        return jsonify({'error': error_msg}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)