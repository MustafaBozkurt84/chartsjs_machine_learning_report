# app.py


import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, make_response,redirect,url_for,render_template,send_file
import jwt
import datetime
from functools import wraps
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import io
from matplotlib.backends.backend_agg import  FigureCanvasAgg as FigureCanvas






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
    url1=None

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
@app.route("/predictionoutput",methods=['GET', 'POST'])
@token_required
def predictionoutput():
    data.token1 = request.args.get('token')
    data.url = ".?token=" + data.token1
    data.url1 = "predictionoutput?token=" + data.token1
    data.url2 = "permutationfeatureimportance?token=" + data.token1
    data.url3 = "correlationfeatureimportance?token=" + data.token1
    data.url4 = "accuracyreport?token=" + data.token1
    data.url5 = "summary?token=" + data.token1
    data.dataframelist = [data.feature_importance, data.permutation_feature_importance, data.prediction_output,
                          data.summary_df, data.accuracy_report]
    for l in data.dataframelist:
        for i in l.columns:
            data.my_dict[i] = list(l[i])
    data.round_list = ["Individual_Feature_Importance", "Pearson_Correlation", "Spearman_Correlation",
                       "Kendall_Correlation", "Accuracy", "Roc_auc_score"]
    for i in data.round_list:
        data.my_dict[i] = [round(i, 2) for i in data.my_dict[i]]
    data.my_dict["summary_df_columns"] = data.summary_df.columns
    data.my_dict["summary_features"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_features_selected"] = request.form.getlist("summary")
    if len(data.my_dict["summary_features_selected"]) == 0:
        data.my_dict["summary_features_selected"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_filtered"] = data.summary_df[
        data.summary_df["Column_Name"].isin(data.my_dict["summary_features_selected"])].reset_index()
    data.my_dict["summary_filtered_len_num"] = len(data.my_dict["summary_filtered"]["Column_Name"])
    data.my_dict["num_summary_df"] = len(data.my_dict["Missing_Percentage"])
    data.my_dict["prediction_columns_"] = data.prediction_output.columns
    data.my_dict["prediction_columns"] = data.my_dict["prediction_columns_"].insert(0, "Select Feature")
    data.my_dict["prediction_output_numeric_columns"] = data.prediction_output.select_dtypes(exclude=['object']).columns
    data.my_dict["prediction_output_numeric_columns"] = data.my_dict["prediction_output_numeric_columns"].insert(0,
                                                                                                                 "Select Feature")
    data.my_dict["prediction_output_categorical_columns"] = data.prediction_output.select_dtypes(
        include=['object']).columns
    data.my_dict["prediction_ids"] = request.form.getlist("predictions")
    data.my_dict["prediction_ids"] = [int(i) for i in data.my_dict["prediction_ids"]]
    if len(data.my_dict["prediction_ids"]) == 0:
        data.my_dict["prediction_ids"] = list(data.prediction_output.loc[1:10, "ID"])
    data.my_dict["prediction_filtered"] = data.prediction_output[
        data.prediction_output["ID"].isin(data.my_dict["prediction_ids"])].reset_index()
    data.my_dict["prediction_filtered_columns"] = data.my_dict["prediction_filtered"].columns
    data.my_dict["prediction_id_column"] = list(data.prediction_output.loc[:, "ID"])
    data.my_dict["prediction_filtered_len_num"] = len(data.my_dict["prediction_filtered"]["ID"])

    data.my_dict["url"] = data.url
    data.my_dict["url1"] = data.url1
    data.my_dict["url2"] = data.url2
    data.my_dict["url3"] = data.url3
    data.my_dict["url4"] = data.url4
    data.my_dict["url5"] = data.url5
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
            data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
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
            data.my_dict_df = data.prediction_output.loc[:, [a, b]].groupby(by=a)[b].mean().reset_index()
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
            data.my_dict_df = data.prediction_output[[a, b]]
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
            # ACCURACY CHART CONFUSİON MATRİX
    for i in range(0, 2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = sns.set_style("darkgrid")

        cf_matrix1 = np.array([[data.accuracy_report["TN"][i], data.accuracy_report["FP"][i]],
                               [data.accuracy_report["FN"][i], data.accuracy_report["TP"][i]]])
        group_names1 = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_counts1 = ["{0:0.0f}".format(value) for value in cf_matrix1.flatten()]
        group_percentages1 = ["{0:.2%}".format(value) for value in cf_matrix1.flatten() / np.sum(cf_matrix1)]
        labels1 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names1, group_counts1, group_percentages1)]
        labels1 = np.asarray(labels1).reshape(2, 2)
        sns.heatmap(cf_matrix1, annot=labels1, fmt="", cmap='Blues')

        fig.savefig(f"./static/confusionmatrix{i}.jpg", transparent=True)

    return render_template('predictionoutput1.html', my_dict=data.my_dict, select_box=data.select_box)



@app.route("/permutationfeatureimportance",methods=['GET', 'POST'])
@token_required
def permutationfeatureimportance():
    data.token1 = request.args.get('token')
    data.url = ".?token=" + data.token1
    data.url1 = "predictionoutput?token=" + data.token1
    data.url2 = "permutationfeatureimportance?token=" + data.token1
    data.url3 = "correlationfeatureimportance?token=" + data.token1
    data.url4 = "accuracyreport?token=" + data.token1
    data.url5 = "summary?token=" + data.token1
    data.dataframelist = [data.feature_importance, data.permutation_feature_importance, data.prediction_output,
                          data.summary_df, data.accuracy_report]
    for l in data.dataframelist:
        for i in l.columns:
            data.my_dict[i] = list(l[i])
    data.round_list = ["Individual_Feature_Importance", "Pearson_Correlation", "Spearman_Correlation",
                       "Kendall_Correlation", "Accuracy", "Roc_auc_score"]
    for i in data.round_list:
        data.my_dict[i] = [round(i, 2) for i in data.my_dict[i]]
    data.my_dict["summary_df_columns"] = data.summary_df.columns
    data.my_dict["summary_features"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_features_selected"] = request.form.getlist("summary")
    if len(data.my_dict["summary_features_selected"]) == 0:
        data.my_dict["summary_features_selected"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_filtered"] = data.summary_df[
        data.summary_df["Column_Name"].isin(data.my_dict["summary_features_selected"])].reset_index()
    data.my_dict["summary_filtered_len_num"] = len(data.my_dict["summary_filtered"]["Column_Name"])
    data.my_dict["num_summary_df"] = len(data.my_dict["Missing_Percentage"])
    data.my_dict["prediction_columns_"] = data.prediction_output.columns
    data.my_dict["prediction_columns"] = data.my_dict["prediction_columns_"].insert(0, "Select Feature")
    data.my_dict["prediction_output_numeric_columns"] = data.prediction_output.select_dtypes(exclude=['object']).columns
    data.my_dict["prediction_output_numeric_columns"] = data.my_dict["prediction_output_numeric_columns"].insert(0,
                                                                                                                 "Select Feature")
    data.my_dict["prediction_output_categorical_columns"] = data.prediction_output.select_dtypes(
        include=['object']).columns
    data.my_dict["prediction_ids"] = request.form.getlist("predictions")
    data.my_dict["prediction_ids"] = [int(i) for i in data.my_dict["prediction_ids"]]
    if len(data.my_dict["prediction_ids"]) == 0:
        data.my_dict["prediction_ids"] = list(data.prediction_output.loc[1:10, "ID"])
    data.my_dict["prediction_filtered"] = data.prediction_output[
        data.prediction_output["ID"].isin(data.my_dict["prediction_ids"])].reset_index()
    data.my_dict["prediction_filtered_columns"] = data.my_dict["prediction_filtered"].columns
    data.my_dict["prediction_id_column"] = list(data.prediction_output.loc[:, "ID"])
    data.my_dict["prediction_filtered_len_num"] = len(data.my_dict["prediction_filtered"]["ID"])

    data.my_dict["url"] = data.url
    data.my_dict["url1"] = data.url1
    data.my_dict["url2"] = data.url2
    data.my_dict["url3"] = data.url3
    data.my_dict["url4"] = data.url4
    data.my_dict["url5"] = data.url5
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
            data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
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
            data.my_dict_df = data.prediction_output.loc[:, [a, b]].groupby(by=a)[b].mean().reset_index()
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
            data.my_dict_df = data.prediction_output[[a, b]]
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
            # ACCURACY CHART CONFUSİON MATRİX
    for i in range(0, 2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = sns.set_style("darkgrid")

        cf_matrix1 = np.array([[data.accuracy_report["TN"][i], data.accuracy_report["FP"][i]],
                               [data.accuracy_report["FN"][i], data.accuracy_report["TP"][i]]])
        group_names1 = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_counts1 = ["{0:0.0f}".format(value) for value in cf_matrix1.flatten()]
        group_percentages1 = ["{0:.2%}".format(value) for value in cf_matrix1.flatten() / np.sum(cf_matrix1)]
        labels1 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names1, group_counts1, group_percentages1)]
        labels1 = np.asarray(labels1).reshape(2, 2)
        sns.heatmap(cf_matrix1, annot=labels1, fmt="", cmap='Blues')

        fig.savefig(f"./static/confusionmatrix{i}.jpg", transparent=True)

    return render_template('permutationfeatureimportance.html', my_dict=data.my_dict, select_box=data.select_box)



@app.route("/correlationfeatureimportance",methods=['GET', 'POST'])
@token_required
def correlationfeatureimportance():
    data.token1 = request.args.get('token')
    data.url = ".?token=" + data.token1
    data.url1 = "predictionoutput?token=" + data.token1
    data.url2 = "permutationfeatureimportance?token=" + data.token1
    data.url3 = "correlationfeatureimportance?token=" + data.token1
    data.url4 = "accuracyreport?token=" + data.token1
    data.url5 = "summary?token=" + data.token1
    data.dataframelist = [data.feature_importance, data.permutation_feature_importance, data.prediction_output,
                          data.summary_df, data.accuracy_report]
    for l in data.dataframelist:
        for i in l.columns:
            data.my_dict[i] = list(l[i])
    data.round_list = ["Individual_Feature_Importance", "Pearson_Correlation", "Spearman_Correlation",
                       "Kendall_Correlation", "Accuracy", "Roc_auc_score"]
    for i in data.round_list:
        data.my_dict[i] = [round(i, 2) for i in data.my_dict[i]]
    data.my_dict["summary_df_columns"] = data.summary_df.columns
    data.my_dict["summary_features"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_features_selected"] = request.form.getlist("summary")
    if len(data.my_dict["summary_features_selected"]) == 0:
        data.my_dict["summary_features_selected"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_filtered"] = data.summary_df[
        data.summary_df["Column_Name"].isin(data.my_dict["summary_features_selected"])].reset_index()
    data.my_dict["summary_filtered_len_num"] = len(data.my_dict["summary_filtered"]["Column_Name"])
    data.my_dict["num_summary_df"] = len(data.my_dict["Missing_Percentage"])
    data.my_dict["prediction_columns_"] = data.prediction_output.columns
    data.my_dict["prediction_columns"] = data.my_dict["prediction_columns_"].insert(0, "Select Feature")
    data.my_dict["prediction_output_numeric_columns"] = data.prediction_output.select_dtypes(exclude=['object']).columns
    data.my_dict["prediction_output_numeric_columns"] = data.my_dict["prediction_output_numeric_columns"].insert(0,
                                                                                                                 "Select Feature")
    data.my_dict["prediction_output_categorical_columns"] = data.prediction_output.select_dtypes(
        include=['object']).columns
    data.my_dict["prediction_ids"] = request.form.getlist("predictions")
    data.my_dict["prediction_ids"] = [int(i) for i in data.my_dict["prediction_ids"]]
    if len(data.my_dict["prediction_ids"]) == 0:
        data.my_dict["prediction_ids"] = list(data.prediction_output.loc[1:10, "ID"])
    data.my_dict["prediction_filtered"] = data.prediction_output[
        data.prediction_output["ID"].isin(data.my_dict["prediction_ids"])].reset_index()
    data.my_dict["prediction_filtered_columns"] = data.my_dict["prediction_filtered"].columns
    data.my_dict["prediction_id_column"] = list(data.prediction_output.loc[:, "ID"])
    data.my_dict["prediction_filtered_len_num"] = len(data.my_dict["prediction_filtered"]["ID"])

    data.my_dict["url"] = data.url
    data.my_dict["url1"] = data.url1
    data.my_dict["url2"] = data.url2
    data.my_dict["url3"] = data.url3
    data.my_dict["url4"] = data.url4
    data.my_dict["url5"] = data.url5
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
            data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
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
            data.my_dict_df = data.prediction_output.loc[:, [a, b]].groupby(by=a)[b].mean().reset_index()
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
            data.my_dict_df = data.prediction_output[[a, b]]
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
            # ACCURACY CHART CONFUSİON MATRİX
    for i in range(0, 2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = sns.set_style("darkgrid")

        cf_matrix1 = np.array([[data.accuracy_report["TN"][i], data.accuracy_report["FP"][i]],
                               [data.accuracy_report["FN"][i], data.accuracy_report["TP"][i]]])
        group_names1 = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_counts1 = ["{0:0.0f}".format(value) for value in cf_matrix1.flatten()]
        group_percentages1 = ["{0:.2%}".format(value) for value in cf_matrix1.flatten() / np.sum(cf_matrix1)]
        labels1 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names1, group_counts1, group_percentages1)]
        labels1 = np.asarray(labels1).reshape(2, 2)
        sns.heatmap(cf_matrix1, annot=labels1, fmt="", cmap='Blues')

        fig.savefig(f"./static/confusionmatrix{i}.jpg", transparent=True)

    return render_template('correlationfeatureimportance.html', my_dict=data.my_dict, select_box=data.select_box)



@app.route("/accuracyreport",methods=['GET', 'POST'])
@token_required
def accuracyreport():
    data.token1 = request.args.get('token')
    data.url = ".?token=" + data.token1
    data.url1 = "predictionoutput?token=" + data.token1
    data.url2 = "permutationfeatureimportance?token=" + data.token1
    data.url3 = "correlationfeatureimportance?token=" + data.token1
    data.url4 = "accuracyreport?token=" + data.token1
    data.url5 = "summary?token=" + data.token1
    data.dataframelist = [data.feature_importance, data.permutation_feature_importance, data.prediction_output,
                          data.summary_df, data.accuracy_report]
    for l in data.dataframelist:
        for i in l.columns:
            data.my_dict[i] = list(l[i])
    data.round_list = ["Individual_Feature_Importance", "Pearson_Correlation", "Spearman_Correlation",
                       "Kendall_Correlation", "Accuracy", "Roc_auc_score"]
    for i in data.round_list:
        data.my_dict[i] = [round(i, 2) for i in data.my_dict[i]]
    data.my_dict["summary_df_columns"] = data.summary_df.columns
    data.my_dict["summary_features"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_features_selected"] = request.form.getlist("summary")
    if len(data.my_dict["summary_features_selected"]) == 0:
        data.my_dict["summary_features_selected"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_filtered"] = data.summary_df[
        data.summary_df["Column_Name"].isin(data.my_dict["summary_features_selected"])].reset_index()
    data.my_dict["summary_filtered_len_num"] = len(data.my_dict["summary_filtered"]["Column_Name"])
    data.my_dict["num_summary_df"] = len(data.my_dict["Missing_Percentage"])
    data.my_dict["prediction_columns_"] = data.prediction_output.columns
    data.my_dict["prediction_columns"] = data.my_dict["prediction_columns_"].insert(0, "Select Feature")
    data.my_dict["prediction_output_numeric_columns"] = data.prediction_output.select_dtypes(exclude=['object']).columns
    data.my_dict["prediction_output_numeric_columns"] = data.my_dict["prediction_output_numeric_columns"].insert(0,
                                                                                                                 "Select Feature")
    data.my_dict["prediction_output_categorical_columns"] = data.prediction_output.select_dtypes(
        include=['object']).columns
    data.my_dict["prediction_ids"] = request.form.getlist("predictions")
    data.my_dict["prediction_ids"] = [int(i) for i in data.my_dict["prediction_ids"]]
    if len(data.my_dict["prediction_ids"]) == 0:
        data.my_dict["prediction_ids"] = list(data.prediction_output.loc[1:10, "ID"])
    data.my_dict["prediction_filtered"] = data.prediction_output[
        data.prediction_output["ID"].isin(data.my_dict["prediction_ids"])].reset_index()
    data.my_dict["prediction_filtered_columns"] = data.my_dict["prediction_filtered"].columns
    data.my_dict["prediction_id_column"] = list(data.prediction_output.loc[:, "ID"])
    data.my_dict["prediction_filtered_len_num"] = len(data.my_dict["prediction_filtered"]["ID"])

    data.my_dict["url"] = data.url
    data.my_dict["url1"] = data.url1
    data.my_dict["url2"] = data.url2
    data.my_dict["url3"] = data.url3
    data.my_dict["url4"] = data.url4
    data.my_dict["url5"] = data.url5
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
            data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
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
            data.my_dict_df = data.prediction_output.loc[:, [a, b]].groupby(by=a)[b].mean().reset_index()
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
            data.my_dict_df = data.prediction_output[[a, b]]
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
            # ACCURACY CHART CONFUSİON MATRİX
    for i in range(0, 2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = sns.set_style("darkgrid")

        cf_matrix1 = np.array([[data.accuracy_report["TN"][i], data.accuracy_report["FP"][i]],
                               [data.accuracy_report["FN"][i], data.accuracy_report["TP"][i]]])
        group_names1 = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_counts1 = ["{0:0.0f}".format(value) for value in cf_matrix1.flatten()]
        group_percentages1 = ["{0:.2%}".format(value) for value in cf_matrix1.flatten() / np.sum(cf_matrix1)]
        labels1 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names1, group_counts1, group_percentages1)]
        labels1 = np.asarray(labels1).reshape(2, 2)
        sns.heatmap(cf_matrix1, annot=labels1, fmt="", cmap='Blues')

        fig.savefig(f"./static/confusionmatrix{i}.jpg", transparent=True)

    return render_template('accuracyreport.html', my_dict=data.my_dict, select_box=data.select_box)


@app.route("/summary",methods=['GET', 'POST'])
@token_required
def summary():
    data.token1 = request.args.get('token')
    data.url = ".?token=" + data.token1
    data.url1 = "predictionoutput?token=" + data.token1
    data.url2 = "permutationfeatureimportance?token=" + data.token1
    data.url3 = "correlationfeatureimportance?token=" + data.token1
    data.url4 = "accuracyreport?token=" + data.token1
    data.url5 = "summary?token=" + data.token1
    data.dataframelist = [data.feature_importance, data.permutation_feature_importance, data.prediction_output,
                          data.summary_df, data.accuracy_report]
    for l in data.dataframelist:
        for i in l.columns:
            data.my_dict[i] = list(l[i])
    data.round_list = ["Individual_Feature_Importance", "Pearson_Correlation", "Spearman_Correlation",
                       "Kendall_Correlation", "Accuracy", "Roc_auc_score"]
    for i in data.round_list:
        data.my_dict[i] = [round(i, 2) for i in data.my_dict[i]]
    data.my_dict["summary_df_columns"] = data.summary_df.columns
    data.my_dict["summary_features"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_features_selected"] = request.form.getlist("summary")
    if len(data.my_dict["summary_features_selected"]) == 0:
        data.my_dict["summary_features_selected"] = list(data.summary_df["Column_Name"])
    data.my_dict["summary_filtered"] = data.summary_df[
        data.summary_df["Column_Name"].isin(data.my_dict["summary_features_selected"])].reset_index()
    data.my_dict["summary_filtered_len_num"] = len(data.my_dict["summary_filtered"]["Column_Name"])
    data.my_dict["num_summary_df"] = len(data.my_dict["Missing_Percentage"])
    data.my_dict["prediction_columns_"] = data.prediction_output.columns
    data.my_dict["prediction_columns"] = data.my_dict["prediction_columns_"].insert(0, "Select Feature")
    data.my_dict["prediction_output_numeric_columns"] = data.prediction_output.select_dtypes(exclude=['object']).columns
    data.my_dict["prediction_output_numeric_columns"] = data.my_dict["prediction_output_numeric_columns"].insert(0,
                                                                                                                 "Select Feature")
    data.my_dict["prediction_output_categorical_columns"] = data.prediction_output.select_dtypes(
        include=['object']).columns
    data.my_dict["prediction_ids"] = request.form.getlist("predictions")
    data.my_dict["prediction_ids"] = [int(i) for i in data.my_dict["prediction_ids"]]
    if len(data.my_dict["prediction_ids"]) == 0:
        data.my_dict["prediction_ids"] = list(data.prediction_output.loc[1:10, "ID"])
    data.my_dict["prediction_filtered"] = data.prediction_output[
        data.prediction_output["ID"].isin(data.my_dict["prediction_ids"])].reset_index()
    data.my_dict["prediction_filtered_columns"] = data.my_dict["prediction_filtered"].columns
    data.my_dict["prediction_id_column"] = list(data.prediction_output.loc[:, "ID"])
    data.my_dict["prediction_filtered_len_num"] = len(data.my_dict["prediction_filtered"]["ID"])

    data.my_dict["url"] = data.url
    data.my_dict["url1"] = data.url1
    data.my_dict["url2"] = data.url2
    data.my_dict["url3"] = data.url3
    data.my_dict["url4"] = data.url4
    data.my_dict["url5"] = data.url5
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
            data.my_dict["my_dict_df_columns"] = data.my_dict_df.columns
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
            data.my_dict_df = data.prediction_output.loc[:, [a, b]].groupby(by=a)[b].mean().reset_index()
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
            data.my_dict_df = data.prediction_output[[a, b]]
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
            # ACCURACY CHART CONFUSİON MATRİX
    for i in range(0, 2):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = sns.set_style("darkgrid")

        cf_matrix1 = np.array([[data.accuracy_report["TN"][i], data.accuracy_report["FP"][i]],
                               [data.accuracy_report["FN"][i], data.accuracy_report["TP"][i]]])
        group_names1 = ["True Neg", "False Pos", "False Neg", "True Pos"]
        group_counts1 = ["{0:0.0f}".format(value) for value in cf_matrix1.flatten()]
        group_percentages1 = ["{0:.2%}".format(value) for value in cf_matrix1.flatten() / np.sum(cf_matrix1)]
        labels1 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names1, group_counts1, group_percentages1)]
        labels1 = np.asarray(labels1).reshape(2, 2)
        sns.heatmap(cf_matrix1, annot=labels1, fmt="", cmap='Blues')

        fig.savefig(f"./static/confusionmatrix{i}.jpg", transparent=True)

    return render_template('summary.html', my_dict=data.my_dict, select_box=data.select_box)

@app.route("/",methods=['GET', 'POST'])
@token_required
def index():
    data.token1 = request.args.get('token')
    data.url = ".?token=" + data.token1
    data.url1 = "predictionoutput?token=" + data.token1
    data.url2 = "permutationfeatureimportance?token=" + data.token1
    data.url3 = "correlationfeatureimportance?token=" + data.token1
    data.url4 = "accuracyreport?token=" + data.token1
    data.url5 = "summary?token=" + data.token1
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



    data.my_dict["url"] = data.url
    data.my_dict["url1"] = data.url1
    data.my_dict["url2"] = data.url2
    data.my_dict["url3"] = data.url3
    data.my_dict["url4"] = data.url4
    data.my_dict["url5"] = data.url5
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
            #ACCURACY CHART CONFUSİON MATRİX
    for i in range(0,2):
            fig, ax = plt.subplots(figsize=(6,6))
            ax = sns.set_style("darkgrid")

            cf_matrix1 = np.array([[data.accuracy_report["TN"][i], data.accuracy_report["FP"][i]],
                                   [data.accuracy_report["FN"][i], data.accuracy_report["TP"][i]]])
            group_names1 = ["True Neg", "False Pos", "False Neg", "True Pos"]
            group_counts1 = ["{0:0.0f}".format(value) for value in cf_matrix1.flatten()]
            group_percentages1 = ["{0:.2%}".format(value) for value in cf_matrix1.flatten() / np.sum(cf_matrix1)]
            labels1 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names1, group_counts1, group_percentages1)]
            labels1 = np.asarray(labels1).reshape(2, 2)
            sns.heatmap(cf_matrix1, annot=labels1, fmt="", cmap='Blues')

            fig.savefig(f"./static/confusionmatrix{i}.jpg", transparent=True)






    return render_template('dashboard1.html', my_dict = data.my_dict ,select_box=data.select_box,url1=data.url1)




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