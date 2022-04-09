# pylint: disable=too-many-arguments
import pandas as pd
from flask import Flask, request,jsonify
import joblib

app=Flask(__name__)

model=joblib.load('KNN Trained and Saved Model.sav')

@app.route('/predictNamed',methods = ['GET','POST'])
def predict_named():
    """This function gets the named args from the api params and predicts the label"""
    v_1=int(request.args.get('v1'))
    v_2=int(request.args.get('v2'))
    v_3=int(request.args.get('v3'))
    v_4=int(request.args.get('v4'))
    v_5=int(request.args.get('v5'))
    v_6=int(request.args.get('v6'))
    v_7=int(request.args.get('v7'))
    prediction = model.predict([[v_1,v_2,v_3,v_4,v_5,v_6,v_7]])
    return jsonify({'Prediction ': int(prediction) })

@app.route('/predictUnnamed/<int:v_1>/<int:v_2>/<int:v_3>/<int:v_4>/<int:v_5>/<int:v_6>/<int:v_7>',methods = ['GET','POST'])
def predict_unnamed(v_1,v_2,v_3,v_4,v_5,v_6,v_7):
    """This function gets the unnamed args from the api and predicts the label"""
    prediction = model.predict([[v_1,v_2,v_3,v_4,v_5,v_6,v_7]])
    return jsonify({'Prediction ': int(prediction) })

@app.route('/predictBodyNamed',methods = ['GET','POST'])
def predict_body_named():
    """This function get the named args from the body of the api and predicts the label"""
    testtext=request.json
    query_df=pd.DataFrame(testtext)
    prediction = model.predict(query_df)
    return jsonify({'Prediction ': int(prediction) })

if __name__ == '__main__':
    app.run(debug=True)
