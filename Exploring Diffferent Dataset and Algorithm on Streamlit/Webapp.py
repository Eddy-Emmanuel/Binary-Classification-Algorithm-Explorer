import numpy as np
import pandas as pd
from io import StringIO

from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import streamlit as st 

# Display title on one line
st.write("### Exploring Binary classification algorithms and datasets")

def GET_DATASET():
    # Add "None" as the initial option
    dataset_ = st.sidebar.selectbox("Select dataset to explore",
                                    ["None", "Breast cancer dataset", "Wine Dataset"])
    
    # Check if the selected dataset is not "None"
    if dataset_ != "None":
        if dataset_ == "Breast cancer dataset":
            X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
        elif dataset_ == "Wine Dataset":
            X, y = datasets.load_wine(return_X_y=True, as_frame=True)

        return pd.concat([X, y], axis="columns"), dataset_
    else:
        # Return None if "None" is selected
        return None, None

def train_model(X, y, X_val, y_val, algo, cv):
    skf = model_selection.StratifiedKFold(n_splits=cv, shuffle=True)
    
    if algo == "LogisticRegression":
        model = LogisticRegression()
    elif algo == "KNeighborsClassifier":
        model = KNeighborsClassifier()
    elif algo == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif algo == "XGBClassifier":
        model = XGBClassifier()
    elif algo == "CatBoostClassifier":
        model = CatBoostClassifier(silent=True)
    else:
        model = LGBMClassifier()

    pred_per_fold = pd.DataFrame()
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        xtrain, xtest = X.iloc[train_idx], X.iloc[test_idx]
        ytrain, ytest = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(xtrain, ytrain)

        if algo == "CatBoostClassifier":
            pred_per_fold[f"fold {idx}"] = model.predict(X_val).flatten()
        else:
            pred_per_fold[f"fold {idx}"] = model.predict(X_val)
    
    return pred_per_fold.mode(axis="columns")[0], model

def Plot_ROC_CURVE(ytrue, ypred):
    fpr, tpr, _ = metrics.roc_curve(ytrue, ypred)
    roc_auc = metrics.auc(fpr, tpr)
    fig, axes = plt.subplots(figsize=(10, 5))
    axes.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    axes.plot([0, 1], [0, 1], ls="--", color="blue")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title("ROC-AUC CURVE")
    plt.legend(loc="lower right")
    st.pyplot(fig)

def plot_feature_importance(feat_imp, label):
    fig, ax = plt.subplots(figsize=(10, 5))
    barplot = sns.barplot(x=feat_imp, y=label, ax=ax)
    barplot.set_xlabel("Importance")
    barplot.set_ylabel("Features")
    barplot.set_title("Feature importance")
    st.pyplot(fig)

def main():
    st.write("#### Data Selection")

    data, data_name = GET_DATASET()

    if data is not None:
        st.write(f"{data_name} dataset")
        st.write(data)

        if st.checkbox("Check data shape"):
            st.write(f"shape: {data.shape}")

        if st.checkbox("Check data info"):
            info_string = StringIO()
            data.info(buf=info_string)
            # Display the info string
            st.text(info_string.getvalue())
        
        if st.checkbox("Check data description"):
            st.write(data.describe().T.drop("count", axis="columns"))
            
        if st.checkbox("Check for missing values"):
            st.write(pd.DataFrame(data.isnull().sum(axis="index"), columns=["No of missing value"]))

        operation = st.sidebar.selectbox(
                                        "Select Machine learning operation",
                                        ["None","EDA", "Modelling"]
                                        )
        if operation != "None":
            if operation == "EDA":
                st.write("#### Exploratory Data Analysis")
                numeric_columns = data.select_dtypes(exclude='O').columns
                cat_columns = data.select_dtypes(include="O").columns

                eda_operation = st.selectbox("Select EDA method", 
                                             ["None", "Scatter plot", "Box plot", "Dist plot", "Violin plot", "Plot correrlation", "Pair plot"])
                
                if eda_operation != "None":
                    if eda_operation == "Plot correrlation":
                        corr = data[numeric_columns].corr()
                        fig, ax = plt.subplots(figsize=(10, 5))
                        heatmap = sns.heatmap(corr, ax=ax)
                        heatmap.set_title("Correlation Matrix")
                        st.pyplot(fig)
                    
                    if eda_operation == "Scatter plot":
                        x_axis = st.selectbox("Select column to plot on x-axis", ["None"]+list(numeric_columns))
                        y_axis = st.selectbox("Select column to plot on y-axis", ["None"]+list(numeric_columns))

                        if (x_axis != "None") & (y_axis != "None"):
                            if (len(x_axis) != 0) & (len(y_axis) != 0):
                                fig, ax = plt.subplots(figsize=(10, 5))
                                scatter_plot = sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
                                scatter_plot.set_title(f"{x_axis.title()} vs {y_axis.title()}")
                                scatter_plot.set_xlabel(x_axis.title())
                                scatter_plot.set_ylabel(y_axis.title())
                                st.pyplot(fig)
                    
                    if eda_operation == "Pair plot":
                        pairplot_col = st.multiselect("Select columns", numeric_columns)
                        if len(pairplot_col) != 0:
                            plt.figure(figsize=(10, 5))
                            pairplot = sns.pairplot(data=data[pairplot_col])
                            st.pyplot(pairplot)

                    if eda_operation == "Box plot":
                        boxplot_feat = st.selectbox("Select column to plot", ["None"]+list(numeric_columns))
                        
                        if boxplot_feat != "None":
                            fig, ax = plt.subplots(figsize=(10, 5))
                            boxplot = sns.boxplot(x=data[boxplot_feat], ax=ax)
                            boxplot.set_title(f"{boxplot_feat.title()}")
                            boxplot.set_xlabel("Values")
                            st.pyplot(fig)
                    
                    if eda_operation == "Violin plot":
                        violinplot_feat = st.selectbox("Select column to plot", ["None"]+list(numeric_columns))
                        if violinplot_feat != "None":
                            fig, ax = plt.subplots(figsize=(10, 5))
                            violinplot = sns.violinplot(x=data[violinplot_feat], ax=ax)
                            violinplot.set_title(f"{violinplot_feat.title()}")
                            violinplot.set_xlabel("Values")
                            st.pyplot(fig)

                    if eda_operation == "Dist plot":
                        distplot_feat = st.selectbox("Select column to plot", ["None"]+list(numeric_columns))
                        if distplot_feat != "None":
                            fig, ax = plt.subplots(figsize=(10, 5))
                            distplot = sns.distplot(x=data[distplot_feat], ax=ax)
                            distplot.set_title(f"{distplot_feat.title()}")
                            distplot.set_xlabel("Values")
                            st.pyplot(fig)
            else:
                st.write(f"#### {operation}")
                X, y = data.drop("target", axis="columns"), data["target"]
                feature_columns = st.multiselect(label="Select columns to train", options=X.columns)

                if len(feature_columns) != 0:
                    algorithms = ["None", "LogisticRegression", "KNeighborsClassifier",
                                "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier",
                                "LGBMClassifier"]
                    
                    algo = st.selectbox("Select a classification algorithm", 
                                        algorithms)
                    
                    if algo != "None":
                        test_size = st.selectbox("Test size", ["None"] + [i/10 for i in range(1, 10)])

                        if test_size != "None":
                            # Preprocess data
                            prep_data = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X[feature_columns]), columns=feature_columns)

                            X_train, X_test, y_train, y_test = model_selection.train_test_split(prep_data, y, test_size=test_size, stratify=y)
                            
                            cv = st.slider("Cross validation (cv)", 2, 50)

                            test_pred, model = train_model(X_train, y_train, X_test, y_test, algo, cv)

                            model_eval = st.selectbox("Model Evaluation", 
                                                    ["None", "Accuracy score", "Confusion Matrix", "Classification report", "ROC AUC CURVE", "Feature Importance"])

                            if model_eval != "None":
                                if model_eval == "Accuracy score":
                                    st.write("Accuracy score: {}".format(metrics.accuracy_score(y_test, test_pred)))

                                elif model_eval == "Confusion Matrix":
                                    cm = metrics.confusion_matrix(y_test, test_pred)
                                    st.text(cm)
                                
                                elif model_eval == "Classification report":
                                    cr = metrics.classification_report(y_test, test_pred)
                                    st.text(cr)

                                elif model_eval == "ROC AUC CURVE":
                                    try:
                                        Plot_ROC_CURVE(y_test, test_pred)
                                    except:
                                        st.warning("Multiclass format is not supported for ROC AUC CURVE")

                                else:
                                    try:
                                        plot_feature_importance(model.feature_importances_, model.feature_names_in_)
                                    except:
                                        try:
                                            plot_feature_importance(model.feature_importances_, model.feature_names_)
                                        except:
                                            try:
                                                plot_feature_importance(model.feature_importances_, X_train.columns)
                                            except:
                                                try:
                                                    plot_feature_importance(np.abs(model.coef_[0]), X_train.columns)
                                                except:
                                                    st.warning("Model has no attribute to feature importance!!!")

    else:
        st.write("No dataset selected.")


if __name__ == "__main__":
    main()