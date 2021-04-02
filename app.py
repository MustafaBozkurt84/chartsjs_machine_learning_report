# app.py


import pandas as pd
import numpy as np

from flask import Flask, jsonify, request, make_response,redirect,url_for,render_template
import jwt
import datetime
from functools import wraps
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns





class DataStore():
    feature_importance = pd.read_csv("feature_importance-2021-03-29-14-28.csv")
    permutation_feature_importance = pd.read_csv("permutation_feature_importance-2021-03-29-14-29.csv")
    prediction_output = pd.read_csv("predictions_output-2021-03-29-14-30.csv")
    summary_df = pd.read_csv("summary_df-2021-03-29-14-27.csv")
    accuracy_report = pd.read_csv("accuracy_report-2021-03-29-14-28.csv")
    my_dict = {}
    dataframelist=None
    round_list=None
    select_box=[]
    select_box1=[]
    url = None
    my_dict_df=None
    a=None
    b=None
    different_ROC_accuracy=None

data = DataStore()



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['SECRET_KEY'] = 'ayhandis'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        data.token = request.args.get('token')

        if not data.token:
            return redirect(url_for('login'))

        try:
            dataa = jwt.decode(data.token, app.config['SECRET_KEY'])

        except:
            return redirect(url_for('login'))

        return f(*args, **kwargs)

    return decorated

@app.after_request
def add_header(r):

    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
@app.route("/",methods=['GET', 'POST'])
@token_required
def index():
    data.token1 = request.args.get('token')
    data.url = ".?token=" + data.token1
    data.dataframelist=[data.feature_importance,data.permutation_feature_importance,data.prediction_output,data.summary_df,data.accuracy_report]
    for l in data.dataframelist:
        for i in l.columns:
             data.my_dict[i]= list(l[i])
    data.round_list=["Individual_Feature_Importance","Pearson_Correlation","Spearman_Correlation","Kendall_Correlation","Accuracy","Roc_auc_score"]
    for i in data.round_list:
        data.my_dict[i]=[round(i,2) for i in data.my_dict[i]]
    data.my_dict["summary_df_columns"]=data.summary_df.columns
    data.my_dict["summary_features"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_features_selected"] = request.form.getlist("summary")
    if len(data.my_dict["summary_features_selected"]) == 0:
        data.my_dict["summary_features_selected"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_filtered"] = data.summary_df[data.summary_df["Column_Name"].isin(data.my_dict["summary_features_selected"])].reset_index()
    data.my_dict["summary_filtered_len_num"] = len(data.my_dict["summary_filtered"]["Column_Name"])
    data.my_dict["num_summary_df"]=len(data.my_dict["Missing_Percentage"])
    data.my_dict["prediction_columns_"] = data.prediction_output.columns
    data.my_dict["prediction_columns"]=data.my_dict["prediction_columns_"].insert(0,"Select Feature")
    data.my_dict["prediction_output_numeric_columns"] = data.prediction_output.select_dtypes(exclude=['object']).columns
    data.my_dict["prediction_output_numeric_columns"]=data.my_dict["prediction_output_numeric_columns"].insert(0,"Select Feature")
    data.my_dict["prediction_output_categorical_columns"] = data.prediction_output.select_dtypes(include=['object']).columns
    data.my_dict["prediction_ids"] = request.form.getlist("predictions")
    data.my_dict["prediction_ids"] = [int(i) for i in data.my_dict["prediction_ids"]]
    if len(data.my_dict["prediction_ids"]) == 0:
        data.my_dict["prediction_ids"] =list(data.prediction_output.loc[1:10,"ID"])
    data.my_dict["prediction_filtered"] = data.prediction_output[data.prediction_output["ID"].isin(data.my_dict["prediction_ids"])].reset_index()
    data.my_dict["prediction_filtered_columns"] = data.my_dict["prediction_filtered"].columns
    data.my_dict["prediction_id_column"] = list(data.prediction_output.loc[:,"ID"])
    data.my_dict["prediction_filtered_len_num"] = len(data.my_dict["prediction_filtered"]["ID"])


    data.my_dict["url"]= data.url
    data.select_box1 = request.form.get("one")
    data.select_box = request.form.get("ones")
    if (data.select_box == None):
        data.select_box = data.my_dict["prediction_output_categorical_columns"][1]
        data.select_box1 = data.my_dict["prediction_output_numeric_columns"][1]

    if data.select_box not in data.my_dict["prediction_output_numeric_columns"]:
        try:

                a = data.select_box
                b = data.select_box1
                data.my_dict["a"] = a
                data.my_dict["b"] = b
                data.my_dict_df = data.prediction_output.groupby(by=a)[b].mean().reset_index()
                data.my_dict["my_dict_df_columns"]=data.my_dict_df.columns
                data.my_dict["create label1"] = list(data.my_dict_df.iloc[:, 0])
                data.my_dict["create chart2"] = list(data.my_dict_df.iloc[:, 1])
                data.my_dict["prediction_output_len"] = len(data.my_dict["create label1"])
                data.my_dict["chart_type"] = "bar"
        except:
            data.select_box = data.my_dict["prediction_output_categorical_columns"][1]
            data.select_box1 = data.my_dict["prediction_output_numeric_columns"][1]
            a = data.select_box
            b = data.select_box1
            data.my_dict["a"] = a
            data.my_dict["b"] = b
            data.my_dict_df = data.prediction_output.loc[:,[a,b]].groupby(by=a)[b].mean().reset_index()
            data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
            data.my_dict["create label1"] = list(data.my_dict_df.iloc[:, 0])
            data.my_dict["create chart2"] = list(data.my_dict_df.iloc[:, 1])
            data.my_dict["prediction_output_len"] = len(data.my_dict["create label1"])
            data.my_dict["chart_type"] = "bar"
    else:
        try:
                a = data.select_box
                b = data.select_box1
                data.my_dict["a"] = a
                data.my_dict["b"] = b
                data.my_dict_df = data.prediction_output[[a,b]]
                data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
                data.my_dict["create label1"] = list(data.my_dict_df.iloc[:, 0])
                data.my_dict["create chart2"] = list(data.my_dict_df.iloc[:, 1])
                data.my_dict["chart_type"] = "line"
                data.my_dict["prediction_output_len"] = len(data.my_dict["create label1"])
        except:
            data.select_box = data.my_dict["prediction_output_categorical_columns"][1]
            data.select_box1 = data.my_dict["prediction_output_numeric_columns"][1]
            a = data.select_box
            b = data.select_box1
            data.my_dict["a"] = a
            data.my_dict["b"] = b
            data.my_dict_df = data.prediction_output[[a, b]]
            data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
            data.my_dict["create label1"] = list(data.my_dict_df.iloc[:, 0])
            data.my_dict["create chart2"] = list(data.my_dict_df.iloc[:, 1])
            data.my_dict["chart_type"] = "line"
            data.my_dict["prediction_output_len"] = len(data.my_dict["create label1"])
            data.my_dict["different_accuracy"]=data.my_dict["Accuray"][0]-data.my_dict["Accuray"][1]









    return render_template('dashboard1.html', my_dict = data.my_dict ,select_box=data.select_box,different_ROC_accuracy=data.different_ROC_accuracy)


@app.route('/login', methods=["POST","GET"])

def login():
    data.auth_dict = {"ayhan": "dis","mustafa":"bozkurt"}
    data.auth = request.authorization
    if data.auth and data.auth.password == data.auth_dict[data.auth.username]:
            data.token_create = jwt.encode({'user': data.auth.username,"password": data.auth.password, 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=86400)},app.config['SECRET_KEY'])
            data.my_token = {'token': data.token_create.decode('utf-8')}

            return redirect(".?token="+data.my_token["token"])



    return make_response('Could not verify!', 401, {'WWW-Authenticate' : 'Basic realm="Login Required"'})


if __name__=='__main__':
    app.run(debug=True)