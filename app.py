# app.py


import pandas as pd
import numpy as np

from flask import Flask, jsonify, request, make_response,redirect,url_for,render_template
import jwt
import datetime
from functools import wraps




class DataStore():
    feature_importance = pd.read_csv("feature_importance-2021-03-29-14-28.csv")
    permutation_feature_importance = pd.read_csv("permutation_feature_importance-2021-03-29-14-29.csv")
    prediction_output = pd.read_csv("predictions_output-2021-03-29-14-30.csv")
    summary_df = pd.read_csv("summary_df-2021-03-29-14-27.csv")
    accuracy_report = pd.read_csv("accuracy_report-2021-03-29-14-28.csv")
    my_dict = {}
    dataframelist=None
    round_list=None
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
    data.round_list=["Individual_Feature_Importance","Pearson_Correlation","Spearman_Correlation","Kendall_Correlation"]
    for i in data.round_list:
        data.my_dict[i]=[round(i,2) for i in data.my_dict[i]]
    data.my_dict["summary_df_columns"]=data.summary_df.columns
    data.my_dict["num_summary_df"]=len(data.my_dict["Missing_Percentage"])

    return render_template('dashboard1.html', my_dict = data.my_dict )


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