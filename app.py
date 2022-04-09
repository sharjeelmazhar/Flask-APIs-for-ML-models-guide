# pylint: disable=too-many-arguments
import pandas as pd
from flask import Flask, request,jsonify
import joblib
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

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
    # pd.to_numeric(query_df)
    # print(query_df)
    v1=query_df['Enter your age?'][0]
    v2=query_df['Gender'][0]
    v3=query_df['Select your Intermediate Field'][0]
    v4=query_df['Enter Father/Guardian Income'][0]
    v5=query_df['1st Preferred Subject'][0]
    v6=query_df['2nd Preferred Subject'][0]
    v7=query_df['Select the Field you like the most'][0]
    v1=int(v1)
    v2=int(v2)
    v3=int(v3)
    v4=int(v4)
    v5=int(v5)
    v6=int(v6)
    v7=int(v7)
    print(v1,v2,v3,v4,v5,v6,v7)
    # prediction = model.predict(query_df)
    prediction = model.predict([[v1,v2,v3,v4,v5,v6,v7]])
    return jsonify({'Prediction ': int(prediction) })

if __name__ == '__main__':
    app.run(debug=True)
