import pickle
import json
import numpy as np
from flask import Flask,jsonify,request
from Insurance_KNN_Project.utils import MedicalInsurance_KNN

app = Flask(__name__)

@app.route("/")
def get_home():
    return "We are at Home API"

@app.route("/Predict_Charges",methods = ['POST','GET'])
def get_medical():
    if request.method == 'POST':
        data = request.form
        age = int(data['age'].strip())
        bmi = eval(data['bmi'].strip())
        children = int(data['children'].strip())
        smoker = data['smoker'].strip().lower()
        region = data['region'].strip().lower()
        # if region == np.nan:  #MIssing value ..if user has not given region 
        #     region = 0

        obj = MedicalInsurance_KNN(age,bmi,children,smoker,region)
        CHARGES = obj.get_charges()

        return jsonify ({'Result':f'Your Medical Insurance Charges are {CHARGES}'})

if __name__ == '__main__':
    app.run()
