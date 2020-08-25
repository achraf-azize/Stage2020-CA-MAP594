from pygam import LinearGAM, LogisticGAM, s, f
import matplotlib.pyplot as plt
import numpy as np
import pandas


class GamExplainer(object):
    def __init__(self,model, X_train, mode, feature_names, numerical_features, link=None, distribution=None,
                 interaction=None):
        self.model = model
        self.X_train = X_train
        self.y_train = model.predict(X_train)
        self.feature_names = feature_names
        self.numerical_features = numerical_features
        self.mode = mode
        self.link = link
        self.distribution = distribution
        self.interaction = interaction
        self._is_fitted = False
        self.explainer = None

    def fit(self):
        S = s(0) if self.feature_names[0] in self.numerical_features else f(0)
        for i in range(1, len(self.feature_names)):
            if self.feature_names[i] in self.numerical_features:
                S += s(i)
            else:
                S += f(i)

        if self.mode == 'regression':
            gam = LinearGAM(S)
            gam.gridsearch(self.X_train, self.y_train)
            self._is_fitted = True
            self.explainer = gam
        elif self.mode == 'classification':
            gam = LogisticGAM(S)
            gam.gridsearch(np.array(self.X_train), self.y_train)
            self._is_fitted = True
            self.explainer = gam
        else:
            raise NameError('ERROR: mode should be regression or classification')

    def plot_dependence(self, list_index_to_draw):
        if not self._is_fitted:
            self.fit()

        plt.figure()
        plt.rcParams['figure.figsize'] = (20, 10)
        fig, axs = plt.subplots(1, len(list_index_to_draw))

        titles = self.feature_names
        for j, i in enumerate(list_index_to_draw):
            ax = axs[j]
            XX = self.explainer.generate_X_grid(term=i)
            ax.plot(XX[:, i], self.explainer.partial_dependence(term=i, X=XX))
            ax.plot(XX[:, i], self.explainer.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
            ax.set_title(titles[i])

    def as_dict(self, list_index_features_to_draw):
        if not self._is_fitted:
            self.fit()
        dict_result = {}
        for i in list_index_features_to_draw:
            XX = self.explainer.generate_X_grid(term=i)
            Y = self.explainer.partial_dependence(term=i, X=XX)
            Y_interval_confiance = self.explainer.partial_dependence(term=i, X=XX, width=.95)[1]
            dict_result[self.feature_names[i]] = {"X": XX[:, i], "Y": Y, "Y_interval_confiance": Y_interval_confiance}
        return dict_result


#### for site web usage
def compute_gam(list_index_features_to_draw, model, data, feature_names, class_names, mode, numerical_features):
    exp_gam = GamExplainer(model=model, data=data,
                             feature_names=feature_names, class_names=class_names,
                            mode=mode, numerical_feature = numerical_features)

    return exp_gam.as_dict(list_index_features_to_draw)
