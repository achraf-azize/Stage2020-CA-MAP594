from sklearn.tree import export_text
from casa_distributed_mlbox.interpret_toolbox.TREPAN import InMemoryModel
from casa_distributed_mlbox.interpret_toolbox.TREPAN import Interpretation
import pandas


class TrepanExplainer(object):
    """
        TrepanExplainer is a module implementing the 'TREPAN' Explanation method, which is a global agnostic interpretability method, that
        approximate globally a black-box model with a tree classifier.

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
    """
    def __init__(self, model, data, feature_names, class_names, mode):
        self.data = data
        #if type(data) != pandas.core.frame.DataFrame:
        #    raise NameError("The input data should be a pandas DataFrame not a {}.format(type(data)")
        self.labels = model.predict(self.data)
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.im_model = InMemoryModel(model.predict_proba, examples=data, target_names=class_names)
        self.explainer = Interpretation(training_data=data, training_labels=self.labels, feature_names=self.feature_names)
        self.surrogate_explainer = self.explainer.tree_surrogate(oracle=self.im_model, seed=42)
        self.is_fitted = False

    def fit(self):
        """
            This method fits the tree surrogate model

        """
        self.surrogate_explainer.fit(self.data, self.labels , use_oracle=True, prune='pre', scorer_type='f1')
        self.is_fitted = True

    def as_text(self):
        """"
            This method returns the tree in a string format

        """

        if not self.is_fitted:
            self.fit()
        text = export_text(self.surrogate_explainer.estimator_, feature_names=list(self.feature_names))
        return text


#### foe site WEB usage
def compute_trepan_string(model, data, feature_names, class_names, mode):
    exp_treepan = TrepanExplainer(model=model, data=data,
                             feature_names=feature_names,class_names=class_names, mode=mode)
    return exp_treepan.as_text()

