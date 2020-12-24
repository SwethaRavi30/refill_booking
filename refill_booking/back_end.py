from flask import Flask,request,url_for,redirect,render_template,jsonify
# from pycaret.regression import*
import pandas as pd
import pickle
import numpy as np
import datetime as dt
app =Flask(__name__,template_folder='template')
model=pickle.load(open('refill_booking_prediction_model.sav','rb'))
cols=['Book Date']
model_week=pickle.load(open('refill_booking_prediction_model_week.sav','rb'))
cols_week=['WeekDate']

@app.route('/')
def home():
    return render_template('front_end.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    data_unseen=pd.DataFrame([int_features],columns=cols)
    data_unseen['Book Date']= pd.to_datetime(data_unseen['Book Date'])
    data_unseen['Book Date']=data_unseen['Book Date'].map(dt.datetime.toordinal)
    prediction=model.predict(data_unseen)    
    return render_template('front_end.html',pred='Expected bookings will be {}'.format(int(prediction)))

@app.route('/predict_week',methods=['POST'])
def predict_week():
    int_features=[x for x in request.form.values()]
    int_features=str(int_features)
    week=int_features[2:6]
    year=int_features[8:10]
    # int_features[1]=int_features[1:]
    temp_df=pd.DataFrame({'year':year,'week':week}, index=[0])
    temp_df['new'] = pd.to_datetime(temp_df.week.astype(str)+temp_df.year.astype(str).add('-1') ,format='%V%G-%u')
    # data_unseen=temp_df['new'].copy()
    # return render_template('front_end.html',pred='Expected bookings will be {}'.format(temp_df))
    # temp_df['new']= pd.to_datetime(temp_df['new'])
    temp_df['new'] = temp_df.apply(lambda row: row['new'] - dt.timedelta(days=row['new'].weekday()), axis=1)
    temp_df['new']=temp_df['new'].map(dt.datetime.toordinal)
    prediction=model_week.predict(np.array(temp_df['new']).reshape(-1,1))    
    return render_template('front_end.html',pred='Expected bookings will be {}'.format(int(prediction)))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

