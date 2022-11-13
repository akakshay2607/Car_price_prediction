from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open('lrmodel.pkl','rb'))

app = Flask(__name__)

cars = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies = sorted(cars['company'].unique())
    car_models = sorted(cars['name'].unique())
    years = sorted(cars['year'].unique(),reverse=True)
    fuel_types = cars['fuel_type'].unique()
    companies.insert(0,'Select car')
    return render_template('index.html',companies=companies,car_models=car_models,years=years,fuel_types=fuel_types)


@app.route('/predict',methods=['post'])
# @cross_origin()

def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')
    
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    
    return str(np.round(prediction[0][0],2))
    
if __name__ == '__main__':
    app.run(debug=True)
    