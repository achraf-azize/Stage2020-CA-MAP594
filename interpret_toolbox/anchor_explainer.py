from anchor import anchor_tabular
import numpy as np


class AnchorExplainer(object):
    """
       AnchorExplainer is a module implementing 'Anchor' Explanation method, which is a local agnostic interpretability method, that
       explains an instance with decision rules.
       Anchor is only working for classification tasks for the moment.

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
       categorical_features: list
           the list of names of 'data' columns that a are categorical
    """
    def __init__(self, model, data, feature_names, class_names=['0', '1'], categorical_features=None):
        self.model = model
        self.data = np.array(data)
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_features = categorical_features
        self.explainer = anchor_tabular.AnchorTabularExplainer(
            self.class_names,
            self.feature_names,
            self.data,)

    def explain(self, idx):
        """
            This method takes an index in the input dataset 'data', and returns the decision rules that explains that instance.

            Parameters
            -----------
            idx: int
               the row index of the element we want to explain in 'data'

            Return
            ----------
            dict: {'Prediction': the prediction of the model ,
                    'Anchor': the decision rules explaining the idx ,
                    'Precision': the precision of the explanation,
                    'Coverage': the percentage of 'data' where the explication holds}

        """
        np.random.seed(1)
        exp = self.explainer.explain_instance(self.data[idx], self.model.predict, threshold=0.95)
        res = {'Prediction': self.model.predict(self.data[idx].reshape(1, -1))[0],
               'Anchor': (' AND '.join(exp.names())), 'Precision': exp.precision(), 'Coverage': exp.coverage()}
        return res


#### for site WEB Usage
def compute_anchor(list_index, model, data, feature_names, class_names, mode):
    exp_anchor = AnchorExplainer(model=model, data=data,
                                 feature_names=feature_names, class_names=class_names, mode = mode)
    dict_res = {}
    for idx in list_index:
        dict_res["idx" + str(idx)] = exp_anchor.explain(idx)

    return dict_res
