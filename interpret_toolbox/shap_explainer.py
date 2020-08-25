import shap
import numpy as np


class ShapExplainer(object):
    """
        ShapExplainer is a module implementing the 'LIME' Explanation method, which is a local agnostic interpretability method, that
        approximate locally a black-box model with a linear model.

        Parameters
        -----------
        model: every object with a predict method
           a predictive model
        data: pandas.DataFrame
           a dataset
        feature_names: list
            the list of names of 'data' columns
        mode: string
            'classification' or 'regression'
        categorical_features: list
           the list of names of 'data' columns that a are categorical
    """

    def __init__(self, model, data, feature_names, mode, algorithm='tree'):
        self.model = model
        self.data = np.vstack(np.array(data)).astype(np.float)
        self.feature_names = feature_names
        self.algorithm = algorithm
        self.mode = mode
        if len(self.data) > 10:
            self.nsamples = 10
        if len(self.data) > 2000:
            self.data = shap.sample(self.data, 2000)
        else:
            self.nsamples = len(self.data)
        if algorithm == 'tree':
            self.explainer = shap.TreeExplainer(model, data=shap.sample(self.data, 50),
                                                feature_perturbation='interventional')
            self.shap_values = self.explainer.shap_values(self.data, check_additivity=False)
        else:
            if self.mode == 'classification':
                self.explainer = shap.KernelExplainer(self.model.predict_proba,
                                                      data=shap.sample(self.data, self.nsamples), link="logit")
            elif self.mode == 'regression':
                self.explainer = shap.KernelExplainer(self.model.predict, data=shap.sample(self.data, self.nsamples))
            self.shap_values = self.explainer.shap_values(self.data, check_additivity=False, nsamples=self.nsamples)

    def summary_plot_all(self):
        if self.mode == 'classification':
            if len(self.shap_values) == 2:
                shap.summary_plot(self.shap_values[1], self.data, feature_names=list(self.feature_names))
            else:
                shap.summary_plot(self.shap_values, self.data, feature_names=list(self.feature_names))
        elif self.mode == 'regression':
            shap.summary_plot(self.shap_values, self.data, feature_names=list(self.feature_names))

    def summary_plot_mean(self):
        shap.summary_plot(self.shap_values, self.data, plot_type="bar", feature_names=list(self.feature_names))

    def force_plot(self, index):
        if self.mode == 'classification':
            if len(self.shap_values) == 2:
                shap.force_plot(self.explainer.expected_value[1], self.shap_values[1][index],
                                feature_names=list(self.feature_names), matplotlib=True)

            else:
                shap.force_plot(self.explainer.expected_value, self.shap_values[index], self.data[index],
                                feature_names=list(self.feature_names),
                                matplotlib=True, link='logit')

        elif self.mode == 'regression':
            shap.force_plot(self.explainer.expected_value, self.shap_values[index], self.data.iloc[index],
                            matplotlib=True, show=False)

        else:
            raise NameError('ERROR: mode should be regression or classification')

    def force_plot_all(self):
        shap.initjs()
        if len(self.shap_values) == 2:
            shap.force_plot(self.explainer.expected_value[1], self.shap_values[1], self.data,
                            feature_names=list(self.feature_names), show=True,
                            matplotlib=False)
        else:
            shap.force_plot(self.explainer.expected_value, self.shap_values, self.data,
                            feature_names=list(self.feature_names), show=True,
                            matplotlib=False)

#### For site WEB usage
# model=self.pipeline["est"]._estimator, data=df_backtest_without_target,
# feature_names=df_backtest_without_target.columns, mode='classification'
def compute_shap(list_index, model, data, feature_names, mode):

    exp_shap = ShapExplainer(model=model, data=data,
                             feature_names=feature_names, mode=mode)
    n_features = len(feature_names)

    labels = model.predict(data)
    if len(exp_shap.shap_values) == 2:
        base_value = exp_shap.explainer.expected_value[1]
    else:
        base_value = exp_shap.explainer.expected_value

    dict_res_total = {'base_value': base_value}
    for i in list_index:
        dict_res_total['idx_' + str(i)] = {}

        if len(exp_shap.shap_values) == 2:
            L = exp_shap.shap_values[1][i]
        else:
            L = exp_shap.shap_values[i]

        index_sort = np.argsort(np.abs(L))
        L_sorted_abs = L[index_sort]
        sum_contribution_feature = [np.sum(L_sorted_abs[:s + 1]) for s in range(n_features)]
        dict_res_total['idx_' + str(i)]['Prediction'] = labels[i]
        for j in range(n_features):
            dict_res_total['idx_' + str(i)][feature_names[n_features - j - 1]] = sum_contribution_feature[
                n_features - j - 1]

    return dict_res_total

