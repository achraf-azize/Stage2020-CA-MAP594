import lime
import lime.lime_tabular
import numpy as np
import pandas as pd


class LimeExplainer(object):
    """
        LimeExplainer is a module implementing the 'LIME' Explanation method, which is a local agnostic interpretability method, that
        approximate locally a black-box model with a linear model.

        Parameters
        -----------
        model: every object with a predict method
           a predictive model
        data: pandas.DataFrame
           a dataset
        feature_names: list
            the list of names of 'data' columns
        class_names: list
           the name of classes in a classification task, default = ['0', '1']
        mode: string
            'classification' or 'regression'
        categorical_features: list
           the list of names of 'data' columns that a are categorical
    """
    def __init__(self, model, data, feature_names, class_names, mode, categorical_features=None):
        self.model = model
        self.data = np.array(data)
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.categorical_features = categorical_features
        self.explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.data),
                                                                feature_names=self.feature_names,
                                                                class_names=self.class_names,
                                                                categorical_features=self.categorical_features,
                                                                verbose=True, mode=self.mode)

    def show_plots(self, index):
        """
            This method takes an index in the input dataset 'data', and returns a local approximate linear model.

            Parameters
            -----------
            idx: int
               the row index of the element we want to explain in 'data'

            Return
            ----------
            pd.DataFrame: 1st column: the region of feature space where the approximation holds true
                        2st column: the coefficient of the linear model
            a summary of explications

        """
        if self.mode == 'classification':
            exp = self.explainer.explain_instance(self.data[index], self.model.predict_proba)
        elif self.mode == 'regression':
            exp = self.explainer.explain_instance(self.data[index], self.model.predict)
        else:
            raise NameError('ERROR: mode should be regression or classification')
        print(pd.DataFrame(exp.as_list()))
        exp.show_in_notebook(show_table=True, show_all=False)


#### For site WEB usage
def compute_lime(list_index, model, data, feature_names, class_names, mode):
    exp_lime = LimeExplainer(model=model, data=data,
                             feature_names=feature_names, class_names=class_names,
                             mode=mode)
    X = exp_lime.data
    dict_res_lime = {}
    for i in list_index:
        exp = exp_lime.explainer.explain_instance(X[i], model.predict_proba)
        S = exp.as_list()
        dict_res_lime["idx_" + str(i)] = {}
        for a in S:
            dict_res_lime["idx_" + str(i)][a[0]] = a[1]

    return dict_res_lime
