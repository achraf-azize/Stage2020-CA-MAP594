from __future__ import division
from functools import partial
from sklearn.metrics.pairwise import cosine_distances
import ast

from sklearn.preprocessing import LabelBinarizer
from sklearn.externals  import six

import abc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.multiclass import type_of_target
import pandas as pd


from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, f1_score
from abc import ABCMeta, abstractmethod


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer



from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import numpy as np
import pydotplus
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

"""Funcs for logging"""
import logging


_CRITICAL = logging.CRITICAL
_ERROR = logging.ERROR
_WARNING = logging.WARNING
_INFO = logging.INFO
_DEBUG = logging.DEBUG
_NOTSET = logging.NOTSET


def build_logger(log_level, logger_name, capture_warning=True):
    logger = logging.Logger(logger_name)

    # All warnings are logged by default
    logging.captureWarnings(capture_warning)

    logger.setLevel(log_level)

    msg_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(msg_formatter)
    stream_handler.setFormatter(msg_formatter)
    logger.addHandler(stream_handler)
    return logger


"Specialized Exceptions for skater"


def exception_factory(exception_name, base_exception=Exception, attributes=None):
    attribute_dict = {
        "__init__": base_exception.__init__
    }
    if isinstance(attributes, dict):
        attributes.update(attributes)
    return type(
        exception_name,
        (base_exception, ),
        attribute_dict
    )


DataSetNotLoadedError = exception_factory('DataSetNotLoadedError')

PartialDependenceError = exception_factory('PartialDependenceError')

FeatureImportanceError = exception_factory('FeatureImportanceError')

DataSetError = exception_factory('DataSetError')

ModelError = exception_factory("ModelError")

TooManyFeaturesError = exception_factory('TooManyFeaturesError',
                                         base_exception=PartialDependenceError)

DuplicateFeaturesError = exception_factory('DuplicateFeaturesError',
                                           base_exception=PartialDependenceError)

EmptyFeatureListError = exception_factory('EmptyFeatureListError',
                                          base_exception=PartialDependenceError)

MalformedGridError = exception_factory("MalformedGridError",
                                       base_exception=PartialDependenceError)

MalformedGridRangeError = exception_factory("MalformedGridRangeError",
                                            base_exception=PartialDependenceError)

MatplotlibUnavailableError = exception_factory('MatplotlibUnavailableError', base_exception=ImportError)

PlotlyUnavailableError = exception_factory('PlotlyUnavailableError', base_exception=ImportError)

TensorflowUnavailableError = exception_factory('TensorflowUnavailableError', base_exception=ImportError)

KerasUnavailableError = exception_factory('KerasUnavailableError', base_exception=ImportError)

MatplotlibDisplayError = exception_factory('MatplotlibDisplayError', base_exception=RuntimeError)

class ModelTypes(object):
    """Stores values for model types and keywords"""
    regressor = 'regressor'
    classifier = 'classifier'
    unknown = 'unknown'

    _valid_ = [regressor, classifier]


class OutputTypes(object):
    """Stores values for output types, and keywords"""
    float = 'float'
    int = 'int'
    string = 'string'
    iterable = 'iterable'
    numeric = 'numeric'
    unknown = 'unknown'

    _valid_ = [float, int, string, iterable, numeric]


class DataTypes(object):

    @staticmethod
    def is_numeric(thing):
        try:
            float(thing)
            return True
        except ValueError:
            return False
        except TypeError:
            return False


    @staticmethod
    def is_string(thing):
        return isinstance(thing, (six.text_type, six.binary_type))


    @staticmethod
    def is_dtype_numeric(dtype):
        assert isinstance(dtype, np.dtype), "expect numpy.dtype, got {}".format(type(dtype))
        return np.issubdtype(dtype, np.number)

    @staticmethod
    def return_data_type(thing):
        """Returns an output type given a variable"""
        if isinstance(thing, (six.text_type, six.binary_type)):
            return StaticTypes.output_types.string
        elif isinstance(thing, int):
            return StaticTypes.output_types.int
        elif isinstance(thing, float):
            return StaticTypes.output_types.float
        elif DataTypes.is_numeric(thing):
            return StaticTypes.output_types.numeric
        elif hasattr(thing, "__iter__"):
            return StaticTypes.output_types.iterable
        else:
            return StaticTypes.unknown


class ScorerTypes(object):
    """Stores values for scorer types"""
    increasing = 'increasing'
    decreasing = 'decreasing'


class StaticTypes(object):
    """Stores values for model types, output types, and keywords"""
    model_types = ModelTypes
    output_types = OutputTypes
    data_types = DataTypes
    scorer_types = ScorerTypes
    unknown = 'unknown'
    not_applicable = 'not applicable'


logger = build_logger(_INFO, __name__)


def flatten(array):
    return [item for sublist in array for item in sublist]


def add_column_numpy_array(array, new_col):
    placeholder = np.ones(array.shape[0])[:, np.newaxis]
    result = np.hstack((array, placeholder))

    if isinstance(new_col, np.ndarray):
        assert array.shape[0] == new_col.shape[0], "input array row counts \
                                                    must be the same. \
                                                    Expected: {0}\
                                                    Actual: {1}".format(array.shape[0],
                                                                        new_col.shape[0])
        assert len(new_col.shape) <= 2, "new column must be 1D or 2D"

        if len(new_col.shape) == 1:
            new_col = new_col[:, np.newaxis]
        return np.hstack((array, new_col))
    elif isinstance(new_col, list):
        assert len(new_col) == array.shape[0], "input array row counts \
                                                    must be the same. \
                                                    Expected: {0}\
                                                    Actual: {1}".format(len(array),
                                                                        len(new_col))
        new_col = np.array(new_col)
        assert len(new_col.shape) == 1, "list elements cannot be iterable"
        new_col = new_col[:, np.newaxis]
        return np.hstack((array, new_col))
    else:
        placeholder = np.ones(array.shape[0])[:, np.newaxis]
        result = np.hstack((array, placeholder))
        result[:, -1] = new_col
        return result


def allocate_samples_to_bins(n_samples, ideal_bin_count=100):
    """goal is as best as possible pick a number of bins
    and per bin samples to a achieve a given number
    of samples.

    Parameters
    ----------

    Returns
    ----------
    number of bins, list of samples per bin
    """

    if n_samples <= ideal_bin_count:
        n_bins = n_samples
        samples_per_bin = [1 for _ in range(n_bins)]
    else:
        n_bins = ideal_bin_count
        remainer = n_samples % ideal_bin_count

        samples_per_bin = np.array([(n_samples - remainer) / ideal_bin_count for _ in range(n_bins)])
        if remainer != 0:
            additional_samples_per_bin = distribute_samples(remainer, n_bins)
            samples_per_bin = samples_per_bin + additional_samples_per_bin
    return n_bins, np.array(samples_per_bin).astype(int)


def distribute_samples(n_samples, n_bins):
    assert n_samples < n_bins, "number of samples should be \
                                less than number of bins"
    space_size = n_bins / n_samples

    samples_per_bin = np.zeros(n_bins).tolist()

    index_counter = 0
    for sample in range(n_samples):
        index = int(index_counter)
        samples_per_bin[index] += 1
        index_counter += space_size
    return np.array(samples_per_bin).astype(int)


def divide_zerosafe(a, b):
    """ diving by zero returns 0 """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


# Lambda for converting data-frame to a dictionary
convert_dataframe_to_dict = lambda key_column_name, value_column_name, df: \
    df.set_index(key_column_name).to_dict()[value_column_name]


def json_validator(json_object):
    """ json validator
    """
    # Reference: https://stackoverflow.com/questions/5508509/how-do-i-check-if-a-string-is-valid-json-in-python
    import json
    try:
        json.loads(json_object)
    except ValueError:
        return False
    return True


def _render_html(file_name, width=None, height=None):
    width, height
    from IPython.core.display import HTML
    return HTML(file_name)


def _render_image(file_name, width=600, height=300):
    from IPython.display import Image
    return Image(file_name, width=width, height=height)


def _render_pdf(file_name, width=600, height=300):
    from IPython.display import IFrame
    IFrame(file_name, width=width, height=height)


def show_in_notebook(file_name_with_type='rendered.html', width=600, height=300, mode=None):
    """ Display generated artifacts(e.g. .png, .html, .jpeg/.jpg) in interactive Jupyter style Notebook

    Parameters
    -----------
    file_name_with_type: str
        specify the name of the file to display
    width: int
        width in pixels to constrain the image
    height: int
        height in pixels to constrain the image
    """
    from IPython.core.display import display, HTML
    if mode is not 'interactive':
        file_type = file_name_with_type.split('/')[-1].split('.')[-1]
        choice_dict = {
            'html': _render_html,
            'png': _render_image,
            'jpeg': _render_image,
            'jpg': _render_image,
            'svg': _render_image,
            'pdf': _render_pdf
        }
        select_type = lambda choice_type: choice_dict[file_type]
        logger.info("File Name: {}".format(file_name_with_type))
        return display(select_type(file_type)(file_name_with_type, width, height))
    else:
        # For now using iframe for some interactive plotting. This should be replaced with a better plotting interface
        iframe_style = '<div style="-webkit-overflow-scrolling:touch; overflow-x:hidden; ' \
                       'overflow-y:auto; width:{}px; height:{}px; margin: -1.2em; ' \
                       '-webkit-transform: scale(0.9) -moz-transform-scale(0.5)"> ' \
                       '<iframe src={} style="width:100%; height:100%; frameborder:1px;">' \
                       '</iframe>' \
                       '</div>'.format(width, height, file_name_with_type)
        return HTML(iframe_style)


class MultiColumnLabelBinarizer(LabelBinarizer):
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output
        self.binarizers = []


    def fit(self, X):
        for x in X.T:
            binarizer = LabelBinarizer()
            binarizer.fit(x)
            self.binarizers.append(binarizer)


    def transform(self, X):
        results = []
        for i, x in enumerate(X.T):
            results.append(self.binarizers[i].transform(x))
        return np.concatenate(results, axis=1)


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


    def inverse_transform(self, X):
        results = []
        column_counter = 0

        for i, binarizer in enumerate(self.binarizers):
            n_cols = binarizer.classes_.shape[0]
            x_subset = X[:, column_counter:column_counter + n_cols]
            inv = binarizer.inverse_transform(x_subset)
            if len(inv.shape) == 1:
                inv = inv[:, np.newaxis]
            results.append(inv)
            column_counter += n_cols
        return np.concatenate(results, axis=1)


__all__ = ['DataManager']


class DataManager(object):
    """Module for passing around data to interpretation objects
    """

    # Todo: we can probably remove some of the keys from data_info, and have properties
    # executed as pure functions for easy to access metadata, such as n_rows, etc

    _n_rows = 'n_rows'
    _dim = 'dim'
    _feature_info = 'feature_info'
    _dtypes = 'dtypes'

    __attribute_keys__ = [_n_rows, _dim, _feature_info, _dtypes]
    __datatypes__ = (pd.DataFrame, pd.Series, np.ndarray)

    def _check_X(self, X):
        if not isinstance(X, self.__datatypes__):
            err_msg = 'Invalid Data: expected data to be a numpy array or pandas dataframe but got ' \
                      '{}'.format(type(X))
            raise(DataSetError(err_msg))
        ndim = len(X.shape)
        self.logger.debug("__init__ data.shape: {}".format(X.shape))

        if ndim == 1:
            X = X[:, np.newaxis]

        elif ndim >= 3:
            err_msg = "Invalid Data: expected data to be 1 or 2 dimensions, " \
                      "Data.shape: {}".format(ndim)
            raise(DataSetError(err_msg))
        return X


    def _check_y(self, y, X):
        """
        convert y to ndarray

        If y is a dataframe:
            return df.values as ndarray
        if y is a series
            return series.values as ndarray
        if y is ndarray:
            return self
        if y is a list:
            return as ndarray
        :param y:
        :param X:
        :return:
        """

        if y is None:
            return None

        assert len(X) == len(y), \
            "len(X) = {0} does not equal len(y) = {1}".format(len(X), len(y))

        if isinstance(y, (pd.DataFrame, pd.Series)):
            return y.values
        elif isinstance(y, np.ndarray):
            return y
        elif isinstance(y, list):
            return np.array(y)
        else:
            raise ValueError("Unrecognized type for y: {}".format(type(y)))


    def __init__(self, X, y=None, feature_names=None, index=None, log_level=30):
        """
        The abstraction around using, accessing, sampling data for interpretation purposes.
        Used by interpretation objects to grab data, collect samples, and handle
        feature names and row indices.

        Parameters
        ----------
            X: 1D/2D numpy array, or pandas DataFrame
                raw data
            y: 1D/2D numpy array, or pandas DataFrame
                ground truth labels for X
            feature_names: iterable of feature names
                Optional keyword containing names of features.
            index: iterable of row names
                Optional keyword containing names of indexes (rows).

        """

        # create logger
        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)

        self.X = self._check_X(X)
        self.y = self._check_y(y, self.X)
        self.data_type = type(self.X)
        self.metastore = None

        self.logger.debug("after transform X.shape: {}".format(self.X.shape))

        if isinstance(self.X, pd.DataFrame):
            if feature_names is None:
                feature_names = self.X.columns.values
            if index is None:
                index = range(self.n_rows)
            self.X.index = index

        elif isinstance(self.X, np.ndarray):
            if feature_names is None:
                feature_names = range(self.X.shape[1])
            if index is None:
                index = range(self.n_rows)

        else:
            raise(ValueError("Invalid: currently we only support {}"
                             "If you would like support for additional data structures let us "
                             "know!".format(self.__datatypes__)))

        self.feature_ids = list(feature_names)
        self.index = list(index)
        self.data_info = {attr: None for attr in self.__attribute_keys__}


    def generate_grid(self, feature_ids, grid_resolution=100, grid_range=(.05, .95)):
        """
        Generates a grid of values on which to compute pdp. For each feature xi, for value
        yj of xi, we will fix xi = yj for every observation in X.

        Parameters
        ----------
            feature_ids(list):
                Feature names for which we'll generate a grid. Must be contained
                by self.feature_ids

            grid_resolution(int):
                The number of unique values to choose for each feature.

            grid_range(tuple):
                The percentile bounds of the grid. For instance, (.05, .95) corresponds to
                the 5th and 95th percentiles, respectively.

        Returns
        -------
        grid(numpy.ndarray): 	There are as many rows as there are feature_ids
                                There are as many columns as specified by grid_resolution
        """

        if not all(i >= 0 and i <= 1 for i in grid_range):
            err_msg = "Grid range values must be between 0 and 1 but got:" \
                      "{}".format(grid_range)
            raise(MalformedGridRangeError(err_msg))

        if not isinstance(grid_resolution, int) and grid_resolution > 0:
            err_msg = "Grid resolution {} is not a positive integer".format(grid_resolution)
            raise(MalformedGridRangeError(err_msg))

        if not all(feature_id in self.feature_ids for feature_id in feature_ids):
            missing_features = []
            for feature_id in feature_ids:
                if feature_id not in self.feature_ids:
                    missing_features.append(feature_id)
            err_msg = "Feature ids {} not found in DataManager.feature_ids".format(missing_features)
            raise(KeyError(err_msg))

        grid_range = [x * 100 for x in grid_range]
        bins = np.linspace(*grid_range, num=grid_resolution).tolist()
        grid = []
        for feature_id in feature_ids:
            data = self[feature_id]
            info = self.feature_info[feature_id]
            # if a feature is categorical (non numeric) or
            # has a small number of unique values, we'll just
            # supply unique values for the grid
            if info['unique'] < grid_resolution or info['numeric'] is False:
                vals = np.unique(data)
            else:
                vals = np.unique(np.percentile(data, bins))
            grid.append(vals)
        grid = np.array(grid)
        grid_shape = [(1, i) for i in [row.shape[0] for row in grid]]
        self.logger.info('Generated grid of shape {}'.format(grid_shape))
        return grid


    def sync_metadata(self):
        self.data_info[self._n_rows] = self.n_rows
        self.data_info[self._dim] = self.dim
        self.data_info[self._dtypes] = self.dtypes
        self.data_info[self._feature_info] = self._calculate_feature_info()


    def _calculate_n_rows(self):
        return self.X.shape[0]


    def _calculate_dim(self):
        return self.X.shape[1]

    @property
    def values(self):
        if self.data_type == pd.DataFrame:
            result = self.X.values
        else:
            result = self.X
        return result


    @property
    def dtypes(self):
        return pd.DataFrame(self.X, columns=self.feature_ids, index=self.index).dtypes


    @property
    def shape(self):
        return self.X.shape


    @property
    def n_rows(self):
        return self.shape[0]


    @property
    def dim(self):
        return self.shape[1]


    def _calculate_feature_info(self):
        feature_info = {}
        for feature in self.feature_ids:
            x = self[feature]
            samples = self.generate_column_sample(feature, n_samples=10)
            samples_are_numeric = map(StaticTypes.data_types.is_numeric, np.array(samples))
            is_numeric = all(samples_are_numeric)
            feature_info[feature] = {
                'type': self.dtypes.loc[feature],
                'unique': len(np.unique(x)),
                'numeric': is_numeric
            }
        return feature_info

    @property
    def feature_info(self):
        if self.data_info[self._feature_info] is None:
            self.data_info[self._feature_info] = self._calculate_feature_info()
        return self.data_info[self._feature_info]


    def _build_metastore(self):

        medians = np.median(self.X, axis=0).reshape(1, self.dim)

        # how far each data point is from the global median
        dists = cosine_distances(self.X, Y=medians).reshape(-1)

        sorted_index = [self.index[i] for i in dists.argsort()]

        return {'sorted_index': sorted_index}

    def __repr__(self):
        return self.X.__repr__()

    def __iter__(self):
        for i in self.feature_ids:
            yield i


    def __setitem__(self, key, newval):
        if issubclass(self.data_type, pd.DataFrame) or issubclass(self.data_type, pd.Series):
            self.__setcolumn_pandas__(key, newval)
        elif issubclass(self.data_type, np.ndarray):
            self.__setcolumn_ndarray__(key, newval)
        else:
            raise ValueError("Can't set item for data of type {}".format(self.data_type))
        self.sync_metadata()


    def __setcolumn_pandas__(self, i, newval):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        self.X[i] = newval


    def __setcolumn_ndarray__(self, i, newval):
        """if you passed in a pandas dataframe, it has columns which are strings."""

        if i in self.feature_ids:
            idx = self.feature_ids.index(i)
            self.X[:, idx] = newval
        else:
            self.X = add_column_numpy_array(self.X, newval)
            self.feature_ids.append(i)


    def __getitem__(self, key):
        if issubclass(self.data_type, pd.DataFrame) or issubclass(self.data_type, pd.Series):
            return self.__getitem_pandas__(key)
        elif issubclass(self.data_type, np.ndarray):
            return self.__getitem_ndarray__(key)
        else:
            raise ValueError("Can't get item for data of type {}".format(self.data_type))


    def __getitem_pandas__(self, i):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        return self.X[i]


    def __getitem_ndarray__(self, i):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        if StaticTypes.data_types.return_data_type(i) == StaticTypes.output_types.iterable:
            idx = [self.feature_ids.index(j) for j in i]
            return self.X[:, idx]
        elif StaticTypes.data_types.is_string(i) or StaticTypes.data_types.is_numeric(i):
            idx = self.feature_ids.index(i)
            return self.X[:, idx]
        else:
            raise(ValueError("Unrecongized index type: {}. This should not happen".format(type(i))))


    def __getrows__(self, idx):
        if self.data_type == pd.DataFrame:
            return self.__getrows_pandas__(idx)
        elif self.data_type == np.ndarray:
            return self.__getrows_ndarray__(idx)
        else:
            raise ValueError("Can't get rows for data of type {}".format(self.data_type))


    def __getrows_pandas__(self, idx):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        if StaticTypes.data_types.return_data_type(idx) == StaticTypes.output_types.iterable:
            i = [self.index.index(i) for i in idx]
        else:
            i = [self.index[idx]]
        return self.X.iloc[i]


    def __getrows_ndarray__(self, idx):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        i = [self.index.index(i) for i in idx]
        return self.X[i]


    def generate_sample(self, sample=True, include_y=False, strategy='random-choice', n_samples=1000,
                        replace=True, bin_count=50):
        """ Method for generating data from the dataset.

        Parameters
        -----------
            sample : boolean
                If False, we'll take the full dataset, otherwise we'll sample.

            include_y: boolean (default=False)

            strategy: string (default='random-choice')
                Supported strategy types 'random-choice', 'uniform-from-percentile', 'uniform-over-similarity-ranks'

            n_samples : int (default=1000)
                Specifies the number of samples to return. Only implemented if strategy is "random-choice".

            replace : boolean (default=True)
                Bool for sampling with or without replacement

            bin_count : int
                If strategy is "uniform-over-similarity-ranks", then this is the number
                of samples to take from each discrete rank.
        """

        __strategy_types__ = ['random-choice', 'uniform-from-percentile', 'uniform-over-similarity-ranks']

        bin_count, samples_per_bin = allocate_samples_to_bins(n_samples, ideal_bin_count=bin_count)
        arg_dict = {
            'sample': sample,
            'strategy': strategy,
            'n_samples': n_samples,
            'replace': replace,
            'samples_per_bin': samples_per_bin,
            'bin_count': bin_count
        }
        self.logger.debug("Generating sample with args:\n {}".format(arg_dict))

        if not sample:
            idx = self.index

        if strategy == 'random-choice':
            idx = np.random.choice(self.index, size=n_samples, replace=replace)

        elif strategy == 'uniform-from-percentile':
            raise(NotImplementedError("We havent coded this yet."))

        elif strategy == 'uniform-over-similarity-ranks':
            sorted_index = self._build_metastore()['sorted_index']
            range_of_indices = list(range(len(sorted_index)))

            def aggregator(samples_per_bin, list_of_indicies):
                n = samples_per_bin[aggregator.count]
                result = str(np.random.choice(list_of_indicies, size=n).tolist())
                aggregator.count += 1
                return result

            aggregator.count = 0
            agg = partial(aggregator, samples_per_bin)

            cuts = pd.qcut(range_of_indices, [i / bin_count for i in range(bin_count + 1)])
            cuts = pd.Series(cuts).reset_index()
            indices = cuts.groupby(0)['index'].aggregate(agg).apply(lambda x: ast.literal_eval(x)).values
            indices = flatten(indices)
            idx = [self.index[i] for i in indices]
        else:
            raise ValueError("Strategy {0} not recognized, currently supported strategies: {1}".format(
                strategy,
                __strategy_types__
            ))
        if include_y:
            return self.__getrows__(idx), self._labels_by_index(idx)
        else:
            return self.__getrows__(idx)


    def generate_column_sample(self, feature_id, *args, **kwargs):
        """Sample a single feature from the data set.

        Parameters
        ----------
        feature_id: hashable
            name of the feature to sample. If no feature names were passed, then
            the features are accessible via their column index.

        """
        dm = DataManager(self[feature_id],
                         feature_names=[feature_id],
                         index=self.index)
        return dm.generate_sample(*args, **kwargs)


    def set_index(self, index):
        self.index = index
        if self.data_type in (pd.DataFrame, pd.Series):
            self.X.index = index


    def _labels_by_index(self, data_index):
        """ Method for grabbing labels associated with given indices.
        """
        # we coerce self.index to a list, so this is fine:
        numeric_index = [self.index.index(i) for i in data_index]

        # do we need to coerce labels to a particular data type?
        return self.y[numeric_index]


    @classmethod
    def _check_input(cls, dataset):
        """
        Ensures that dataset is pandas dataframe, and dataset is not empty
        :param dataset: skater.__datatypes__
        :return:
        """
        if not isinstance(dataset, (pd.DataFrame)):
            err_msg = "dataset must be a pandas.DataFrame"
            raise DataSetError(err_msg)

        if len(dataset) == 0:
            err_msg = "dataset is empty"
            raise DataSetError(err_msg)


"""Model class."""


class Scorer(object):
    """
    Base Class for all skater scoring functions.
    Any Scoring function must consume a model.
    Any scorer must determine the types of models that are compatible.

    """

    __metaclass__ = ABCMeta


    model_types = None
    prediction_types = None
    label_types = None

    def __init__(self, model):
        self.model = model

    @classmethod
    def check_params(cls):
        assert all([i in StaticTypes.model_types._valid_ for i in cls.model_types])
        assert all([i in StaticTypes.output_types._valid_ for i in cls.prediction_types])
        assert all([i in StaticTypes.output_types._valid_ for i in cls.label_types])


    @classmethod
    def check_model(cls, model):

        assert model.model_type in cls.model_types, "Scorer {0} not valid for models of type {1}, " \
                                                    "only {2}".format(cls,
                                                                      model.model_type,
                                                                      cls.model_types)


    def __call__(self, y_true, y_predicted, sample_weight=None):
        self.check_model(self.model)
        self.check_data(y_true, y_predicted)
        # formatted_y = self.model.transformer(self.model.output_formatter(y_true))
        return self._score(y_true, y_predicted, sample_weight=sample_weight)


    @staticmethod
    @abstractmethod
    def check_data(y_true, y_predicted):
        pass


class RegressionScorer(Scorer):
    model_types = [StaticTypes.model_types.regressor]
    prediction_types = [
        StaticTypes.output_types.numeric,
        StaticTypes.output_types.float,
        StaticTypes.output_types.int
    ]
    label_types = [
        StaticTypes.output_types.numeric,
        StaticTypes.output_types.float,
        StaticTypes.output_types.int
    ]

    @staticmethod
    def check_data(y_true, y_predicted):
        assert hasattr(y_predicted, 'shape'), \
            'outputs must have a shape attribute'
        assert hasattr(y_true, 'shape'), \
            'y_true must have a shape attribute'
        assert (len(y_predicted.shape) == 1) or (y_predicted.shape[1] == 1), \
            "Regression outputs must be 1D, " \
            "got {}".format(y_predicted.shape)
        assert (len(y_true.shape) == 1) or (y_true.shape[1] == 1), \
            "Regression outputs must be 1D, " \
            "got {}".format(y_true.shape)


# Regression Scorers
class MeanSquaredError(RegressionScorer):
    __name__ = "MSE"
    type = StaticTypes.scorer_types.decreasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        return mean_squared_error(y_true, y_predicted, sample_weight=sample_weight)


class MeanAbsoluteError(RegressionScorer):
    __name__ = "MAE"
    type = StaticTypes.scorer_types.decreasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        return mean_absolute_error(y_true, y_predicted, sample_weight=sample_weight)


class RSquared(RegressionScorer):
    __name__ = "R2"
    # Reference: https://en.wikipedia.org/wiki/Coefficient_of_determination
    # The score values range between [0, 1]. The best possible value is 1, however one could expect negative values as
    # well because of the arbitrary model fit.
    type = StaticTypes.scorer_types.increasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        return r2_score(y_true, y_predicted, sample_weight=sample_weight)


class ClassifierScorer(Scorer):

    """
    * predictions must be N x K matrix with N rows and K classes.
    * labels must be be N x K matrix with N rows and K classes.
    """

    model_types = [StaticTypes.model_types.classifier]
    prediction_types = [StaticTypes.output_types.numeric, StaticTypes.output_types.float, StaticTypes.output_types.int]
    label_types = [StaticTypes.output_types.numeric, StaticTypes.output_types.float, StaticTypes.output_types.int]

    @staticmethod
    def check_data(y_true, y_predicted):
        assert hasattr(y_predicted, 'shape'), 'outputs must have a shape attribute'
        assert hasattr(y_true, 'shape'), 'y_true must have a shape attribute'


# Metrics related to Classification
class CrossEntropy(ClassifierScorer):
    __name__ = "cross-entropy"
    type = StaticTypes.scorer_types.decreasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        """

        :param X: Dense X of probabilities, or binary indicator
        :param y:
        :param sample_weights:
        :return:
        """
        return log_loss(y_true, y_predicted, sample_weight=sample_weight)


class F1(ClassifierScorer):
    __name__ = "f1-score"
    type = StaticTypes.scorer_types.increasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None, average='weighted'):
        """

        :param X: Dense X of probabilities, or binary indicator
        :param y: indicator
        :param sample_weights:
        :return:
        """
        if len(y_predicted.shape) == 2:
            preds = y_predicted.argmax(axis=1)
        else:
            preds = y_predicted

        return f1_score(y_true, preds, sample_weight=sample_weight, average=average)


class ScorerFactory(object):
    """
    The idea is that we initialize the object with the model,
    but also provide an api for retrieving a static scoring function
    after checking that things are ok.
    """
    def __init__(self, model):
        if model.model_type == StaticTypes.model_types.regressor:
            self.mse = MeanSquaredError(model)
            self.mae = MeanAbsoluteError(model)
            self.r2 = RSquared(model)
            self.default = self.mae
        elif model.model_type == StaticTypes.model_types.classifier:
            self.cross_entropy = CrossEntropy(model)
            self.f1 = F1(model)
            # TODO: Not sure why the first condition is relevant, probably some early design decision.
            # TODO Need to check other examples and add more test before removing the first check completely
            if model.probability is not None and not 'unknown' or model.probability is True:
                self.default = self.cross_entropy
            else:
                self.default = self.f1

        self.type = self.default.type


    def __call__(self, y_true, y_predicted, sample_weight=None):
        return self.default(y_true, y_predicted, sample_weight=sample_weight)


    def get_scorer_function(self, scorer_type='default'):
        """
        Returns a scoring function as a pure function.

        Parameters
        ----------

        scorer_type: string
            Specifies which scorer to use. Default value 'default' returns f1 for classifiers that return labels,
            cross_entropy for classifiers that return probabilities, and mean absolute error for regressors.


        Returns
        -------
            .score staticmethod of skater.model.scorer.Scorer object.
        """
        assert scorer_type in self.__dict__, "Scorer type {} not recognized " \
                                             "or allowed for model type".format(scorer_type)
        scorer = self.__dict__[scorer_type]._score
        scorer.type = self.__dict__[scorer_type].type
        scorer.name = self.__dict__[scorer_type].__name__
        return scorer



class ModelType(object):
    """What is a model? A model needs to make predictions, so a means of
    passing data into the model, and receiving results.

    Goals:
        We want to abstract away how we access the model.
        We want to make inferences about the format of the output.
        We want to able to map model outputs to some smaller, universal set of output types.
        We want to infer whether the model is real valued, or classification (n classes?)

    # Todos:
    * check unique_vals are unique
    * check that if probability=False, predictions arent continuous

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 log_level=30,
                 target_names=None,
                 examples=None,
                 feature_names=None,
                 unique_values=None,
                 input_formatter=None,
                 output_formatter=None,
                 model_type=None,
                 probability=None):
        """
        Base model class for wrapping prediction functions. Common methods
        involve output type inference in requiring predict methods

        Parameters
        ----------
            log_level: int
                0, 10, 20, 30, 40, or 50 for verbosity of logs.
            target_names: arraytype
                The names of the target variable/classes. There should be as many
                 names as there are outputs per prediction (n=1 for regression,
                 n=2 for binary classification, etc). Defaults to Predicted Value for
                 regression and Class 1...n for classification.



        Attributes
        ----------
            model_type: string

        """

        self._check_model_type(model_type)
        self._check_probability(probability)

        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)
        self.examples = None

        self.output_var_type = StaticTypes.unknown
        self.output_shape = StaticTypes.unknown
        self.n_classes = StaticTypes.unknown
        self.input_shape = StaticTypes.unknown
        if probability is None:
            self.probability = StaticTypes.unknown
        else:
            self.probability = probability

        if model_type is None:
            self.model_type = StaticTypes.unknown
        else:
            self.model_type = model_type

        self.transformer = identity_function
        self.label_encoder = LabelEncoder()
        self.target_names = target_names
        self.feature_names = feature_names
        self.unique_values = unique_values
        self.input_formatter = input_formatter or identity_function
        self.output_formatter = output_formatter or identity_function

        self.has_metadata = False

        if examples is not None:
            self.input_type = type(examples)
            examples = DataManager(examples, feature_names=feature_names)
            self._build_model_metadata(examples)
        else:
            self.input_type = None
            self.logger.warn("No examples provided, cannot infer model type")


    def _check_model_type(self, model_type):
        __types__ = [None,
                     StaticTypes.model_types.classifier,
                     StaticTypes.model_types.regressor]
        assert model_type in __types__, \
            "Expected model_type {0}, got {1}".format(__types__, model_type)


    def _check_probability(self, probability):
        __types__ = [None, True, False]
        assert probability in __types__, \
            "Expected model_type {0}, got {1}".format(__types__, probability)


    def predict(self, *args, **kwargs):
        """
        The way in which the submodule predicts values given an input
        """
        if self.has_metadata is False:
            self.has_metadata = True
            examples = DataManager(*args)
            self._build_model_metadata(examples)
        return self.transformer(self.output_formatter(self._execute(self.input_formatter(*args, **kwargs))))


    @property
    def scorers(self):
        """

        :param X:
        :param y:
        :param sample_weights:
        :param scorer:
        :return:
        """
        # TODO: This is a temporary work around. I feel that Scorer component needs to be designed slightly different.
        # Ideally, with the initialization of the Interpretation instance we want to get Interpretation Algorithms all
        # ready to go. Similary, when we initialize InMemory model/Deploy model,
        # we should be able to specifying the evaluation type but giving user the freedom to specify the metric
        # later as well. The below, condition is commented to implement TreeSurrogate.
        # if not self.has_metadata:
        #     raise NotImplementedError("The model needs metadata before "
        #                               "the scorer can be used. Please first"
        #                               "run model.predict(X) on a couple examples"
        #                               "first")
        # else:
        scorers = ScorerFactory(self)
        return scorers

    @abc.abstractmethod
    def _execute(self, *args, **kwargs):
        """
        The way in which the submodule predicts values given an input
        """
        return


    @abc.abstractmethod
    def _predict(self, *args, **kwargs):
        """
        The way in which the submodule predicts values given an input
        """
        return

    @abc.abstractmethod
    def _get_static_predictor(self, *args, **kwargs):
        """Return a static prediction function to avoid shared state in multiprocessing"""
        return


    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


    def check_examples(self, examples):
        """
        Ties examples to self. equivalent to self.examples = np.array(examples).
        Parameters
        ----------
        examples: array type

        """
        if isinstance(examples, (pd.DataFrame, np.ndarray)):
            return examples
        else:
            return np.array(examples)


    def _if_no_prob(self, value):
        if self.probability == StaticTypes.unknown:
            return value
        else:
            return self.probability


    def _if_no_model(self, value):
        if self.model_type == StaticTypes.unknown:
            return value
        else:
            return self.model_type


    def _build_model_metadata(self, dataset):
        """
        Determines the model_type, output_type. Side effects
        of this method are to mutate object's attributes (model_type,
        n_classes, etc).

        Parameters
        ----------
        examples: pandas.DataFrame or numpy.ndarray
            The examples that will be passed through the predict function.
            The outputs from these examples will be used to make inferences
            about the types of outputs the function generally makes.

        """
        self.logger.debug("Beginning output checks")

        if self.input_type in (pd.DataFrame, None):
            outputs = self.predict(dataset.X)
        elif self.input_type == np.ndarray:
            outputs = self.predict(dataset.X)
        else:
            raise ValueError("Unrecognized input type: {}".format(self.input_type))

        self.input_shape = dataset.X.shape
        self.output_shape = outputs.shape

        ndim = len(outputs.shape)
        if ndim > 2:
            raise(ValueError("Unsupported model type, output dim = {}".format(ndim)))

        try:
            # continuous, binary, continuous multioutput, multiclass, multilabel-indicator
            self.output_type = type_of_target(outputs)
        except:
            self.output_type = False

        if self.output_type == 'continuous':
            # 1D array of continuous values
            self.model_type = self._if_no_model(StaticTypes.model_types.regressor)
            self.n_classes = 1
            self.probability = self._if_no_prob(False)

        elif self.output_type == 'multiclass':
            # 2D array of 1s and 0, exclusive
            self.model_type = self._if_no_model(StaticTypes.model_types.classifier)
            self.probability = self._if_no_prob(False)
            self.n_classes = len(np.unique(outputs))

        elif self.output_type == 'continuous-multioutput':
            # 2D array of continuous values
            self.model_type = self._if_no_model(StaticTypes.model_types.classifier)
            self.probability = self._if_no_prob(True)
            self.n_classes = outputs.shape[1]

        elif self.output_type == 'binary':
            # 2D array of 1s and 0, non exclusive
            self.model_type = self._if_no_model(StaticTypes.model_types.classifier)
            self.probability = self._if_no_prob(False)
            self.n_classes = 2

        elif self.output_type == 'multilabel-indicator':
            # 2D array of 1s and 0, non exclusive
            self.model_type = self._if_no_model(StaticTypes.model_types.classifier)
            self.probability = self._if_no_prob(False)
            self.n_classes = outputs.shape[1]

            if self.probability:
                self.output_type = 'continuous-multioutput'

        else:
            err_msg = "Could not infer model type"
            self.logger.debug("Inputs: {}".format(dataset.X))
            self.logger.debug("Outputs: {}".format(outputs))
            self.logger.debug("sklearn response: {}".format(self.output_type))
            ModelError(err_msg)

        if self.target_names is None:
            self.target_names = ["predicted_{}".format(i) for i in range(self.n_classes)]

        if self.unique_values is None and self.model_type == 'classifier' and self.probability is False:
            raise (ModelError('If using classifier without probability scores, unique_values cannot '
                                         'be None'))

        self.transformer = self.transformer_func_factory(outputs)

        reports = self.model_report(dataset.X)
        for report in reports:
            self.logger.debug(report)

        self.has_metadata = True


    def transformer_func_factory(self, outputs):
        """
        In the event that the predict func returns 1D array of predictions,
        then this returns a formatter to convert outputs to a 2D one hot encoded
        array.

        For instance, if:
            predict_fn(data) -> ['apple','banana']
        then
            transformer = Model.transformer_func_factory()
            transformer(predict_fn(data)) -> [[1, 0], [0, 1]]

        Returns
        ----------
        (callable):
            formatter function to wrap around predict_fn
        """

        # Note this expression below assumptions (not probability) evaluates to false if
        # and only if the model does not return probabilities. If unknown, should be true
        if self.model_type == StaticTypes.model_types.classifier and not self.probability:
            # fit label encoder
            # unique_values could ints/strings, etc.
            artificial_samples = np.array(self.unique_values)
            self.logger.debug("Label encoder fit on examples of shape: {}".format(outputs.shape))

            def check_classes(classes):
                if len(classes) > 2:
                    return classes
                elif len(classes) == 2:
                    # to get 2 columns from label_binarize, we need
                    # to pretend we have at least 3 classes.
                    # adding will preserve type of underlying classes
                    fake_class = classes[0] + classes[1]
                    return np.concatenate((classes, np.array([fake_class])))
                else:
                    raise ValueError("Less than 2 classes found in unique_classes")

            # defining this as a closure so it can be executed
            # outside the class
            def transformer(output):
                # numeric index of original classes.
                idx = list(range(len(artificial_samples)))
                classes = check_classes(artificial_samples)
                return label_binarize(output, classes)[:, idx]
            return transformer
        else:
            return identity_function


    def model_report(self, examples):
        """
        Just returns a list of model attributes as a list

        Parameters
        ----------
        examples: array type:
            Examples to use for which we report behavior of predict_fn.


        Returns
        ----------
        reports: list of strings
            metadata about function.

        """
        examples = DataManager(examples, feature_names=self.feature_names)
        reports = []
        if isinstance(self.examples, np.ndarray):
            raw_predictions = self.predict(examples)
            reports.append("Example: {} \n".format(examples[0]))
            reports.append("Outputs: {} \n".format(raw_predictions[0]))
        reports.append("Model type: {} \n".format(self.model_type))
        reports.append("Output Var Type: {} \n".format(self.output_var_type))
        reports.append("Output Shape: {} \n".format(self.output_shape))
        reports.append("N Classes: {} \n".format(self.n_classes))
        reports.append("Input Shape: {} \n".format(self.input_shape))
        reports.append("Probability: {} \n".format(self.probability))
        return reports

    def predict_subset_classes(self, data, subset_of_classes):
        """Filters predictions to a subset of classes."""
        if subset_of_classes is None:
            return self.predict(data)
        else:
            return DataManager(self.predict(data), feature_names=self.target_names)[subset_of_classes].X


def identity_function(x):
    return x




# reference: http://wingraphviz.sourceforge.net/wingraphviz/language/colorname.htm
# TODO: Make the color scheme for regression and classification homogeneous
color_schemes = ['aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
                 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
                 'cornsilk', 'crimson', 'cyan', 'darkgoldenrod', 'darkgreen', 'darkkhaki', 'darkolivegreen', 'darkorange',
                 'darkorchid', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey',
                 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
                 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
                 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
                 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrod',
                 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen',
                 'lightskyblue', 'lightslateblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow',
                 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
                 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
                 'navyblue', 'oldlace', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
                 'palevioletred', 'papayawhip', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
                 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue',
                 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato',
                 'turquoise', 'violet', 'violetred', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']


# Reference: https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/tree/_tree.pyx
TREE_LEAF = -1
TREE_UNDEFINED = -2


def _get_colors(num_classes, random_state=1):
    np.random.seed(random_state)
    color_index = np.random.randint(0, len(color_schemes), num_classes)
    colors = np.array(color_schemes)[color_index]
    return colors


def _generate_graph(est, est_type='classifier', classes=None, features=None,
                    enable_node_id=True, coverage=True):
    dot_data = StringIO()
    # class names are needed only for "Classification" for "Regression" it is set to None
    c_n = classes if est_type == 'classifier' else None
    export_graphviz(est, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=features,
                    class_names=c_n, node_ids=enable_node_id, proportion=coverage)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph


def _set_node_properites(estimator, estimator_type, graph_instance, color_names, default_color):
    # Query and assign properties to each node
    thresholds = estimator.tree_.threshold
    values = estimator.tree_.value
    left_node = estimator.tree_.children_left
    right_node = estimator.tree_.children_right

    nodes = graph_instance.get_node_list()
    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            if estimator_type == 'classifier':
                value = values[int(node.get_name())][0]
                # 1. Color only the leaf nodes, where one class is dominant or if it is a leaf node
                # 2. For mixed population or otherwise set the default color
                if max(value) == sum(value) or thresholds[int(node.get_name())] == TREE_UNDEFINED or \
                        left_node[int(node.get_name())] and right_node[int(node.get_name())] == TREE_LEAF:
                    node.set_fillcolor(color_names[np.argmax(value)])
                else:
                    node.set_fillcolor(default_color)
            else:
                # if the estimator type is a "regressor", then the intensity of the color is defined by the
                # population coverage for a particular value
                percent = estimator.tree_.n_node_samples[int(node.get_name())] / float(estimator.tree_.n_node_samples[0])
                rgba = plt.cm.get_cmap(color_names)(percent)
                hex_code = rgb2hex(rgba)
                node.set_fillcolor(hex_code)
                graph_instance.set_colorscheme(color_names)
    return graph_instance


# https://stackoverflow.com/questions/48085315/interpreting-graphviz-output-for-decision-tree-regression
# https://stackoverflow.com/questions/42891148/changing-colors-for-decision-tree-plot-created-using-export-graphviz
# Color scheme info: http://wingraphviz.sourceforge.net/wingraphviz/language/colorname.htm
# Currently, supported only for sklearn models
def plot_tree(estimator, estimator_type='classifier', feature_names=None, class_names=None, color_list=None,
              colormap_reg='PuBuGn', enable_node_id=True, coverage=True, seed=2):

    graph = _generate_graph(estimator, estimator_type, class_names, feature_names, enable_node_id, coverage)

    if estimator_type == 'classifier':
        # if color is not assigned, pick color uniformly random from the color list defined above if the estimator
        # type is "classification"
        colors = color_list if color_list is not None else _get_colors(len(class_names), seed)
        default_color = 'cornsilk'
    else:
        colors = colormap_reg
        default_color = None

    graph = _set_node_properites(estimator, estimator_type, graph, color_names=colors, default_color=default_color)

    # Set the color scheme for the edges
    edges = graph.get_edge_list()
    for ed in edges:
        ed.set_color('steelblue')
    return graph


_return_value = lambda estimator_type, v: 'Predicted Label: {}'.format(str(np.argmax(v))) \
    if estimator_type == 'classifier' else 'Value: {}'.format(str(v))


def _global_decisions_as_txt(est_type, label_color, criteria_color, if_else_color, values,
                             features, thresholds, l_nodes, r_nodes):
    # define "if and else" string patterns for extracting the decision rules
    if_str_pattern = lambda offset, node: offset + "if {}{}".format(criteria_color, features[node]) \
        + " <= {}".format(str(thresholds[node])) + if_else_color + " {"

    other_str_pattern = lambda offset, str_type: offset + if_else_color + str_type

    def _recurse_tree(left_node, right_node, threshold, node, depth=0):
        offset = "  " * depth
        if threshold[node] != TREE_UNDEFINED:
            print(if_str_pattern(offset, node))
            if left_node[node] != TREE_LEAF:
                _recurse_tree(left_node, right_node, threshold, left_node[node], depth + 1)
                print(other_str_pattern(offset, "} else {"))
                if right_node[node] != TREE_LEAF:
                    _recurse_tree(left_node, right_node, threshold, right_node[node], depth + 1)
                print(other_str_pattern(offset, "}"))
        else:
            print(offset, label_color, _return_value(est_type, values[node]))

    _recurse_tree(l_nodes, r_nodes, thresholds, 0)


def _local_decisions_as_txt(est, est_type, label_color, criteria_color, if_else_color,
                            values, features, thresholds, input_X):
    greater_or_less = lambda f_v, s_c: "<=" if f_v <= s_c else ">"
    as_str_pattern = lambda offset, node_id, \
        feature_value, sign: offset + \
        "As {}{}{}".format(criteria_color, features[node_id], "[" + str(feature_value) + "]") + \
        " {} {}".format(sign, str(thresholds[node_id])) + if_else_color + " then,"

    path = est.decision_path(input_X.values.reshape(1, -1))
    node_indexes = path.indices
    leaf_id = est.apply(input_X.values.reshape(1, -1))
    depth = 0
    for node_index in node_indexes:
        offset = "  " * depth
        if leaf_id != node_index:
            feature_value = input_X[features[node_index]]
            print(as_str_pattern(offset, node_index, feature_value,
                                 greater_or_less(feature_value, thresholds[node_index])))
            depth += 1
        else:
            print(offset, label_color, _return_value(est_type, values[node_index]))


# Current implementation is specific to sklearn models.
# Reference: https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
# TODO: Figure out ways to make it generic for other frameworks
def tree_to_text(tree, feature_names, estimator_type='classifier', scope='global', X=None):
    # defining colors
    label_value_color = "\033[1;34;49m"  # blue
    split_criteria_color = "\033[0;32;49m"  # green
    if_else_quotes_color = "\033[0;30;49m"  # if and else quotes

    left_nodes = tree.tree_.children_left
    right_nodes = tree.tree_.children_right
    criterias = tree.tree_.threshold
    feature_names = [feature_names[i] for i in tree.tree_.feature]
    values = tree.tree_.value

    if scope == "global":
        return _global_decisions_as_txt(estimator_type, label_value_color, split_criteria_color,
                                        if_else_quotes_color, values, feature_names, criterias, left_nodes, right_nodes)
    else:
        return _local_decisions_as_txt(tree, estimator_type, label_value_color, split_criteria_color,
                                       if_else_quotes_color, values, feature_names, criterias, X)




class TreeSurrogate(object):
    """ :: Experimental :: The implementation is currently experimental and might change in future.
    The idea of using TreeSurrogates as means for explaining a model's(Oracle or the base model)
    learned decision policies(for inductive learning tasks) is inspired by the work of Mark W. Craven
    described as the TREPAN algorithm. In this explanation learning hypothesis, the base estimator(Oracle)
    could be any form of supervised learning predictive models. The explanations are approximated using
    DecisionTrees(both for Classification/Regression) by learning decision boundaries similar to that learned by
    the Oracle(predictions from the base model are used for learning the DecisionTree representation).
    The implementation also generates a fidelity score to quantify tree based surrogate model's
    approximation to the Oracle. Ideally, the score should be 0 for truthful explanation
    both globally and locally.

    Parameters
    ----------
    oracle : InMemory instance type
        model instance having access to the base estimator(InMemory/DeployedModel).
        Currently, only InMemory is supported.
    splitter : str (default="best")
        Strategy used to split at each the node. Supported strategies("best" or "random").
    max_depth : int (default=None)
        Defines the maximum depth of a tree. If 'None' then nodes are expanded till all leaves are \
        pure or contain less than min_samples_split samples.
        Deeper trees are prone to be more expensive and tend to over-fit.
        Pruning is a technique which could be applied to avoid over-fitting.
    min_samples_split : int/float (default=2)
        Defines the minimum number of samples required to split an internal node:

        - int, specifies the minimum number of samples
        - float, then represents a percentage. Minimum number of samples is computed as \
          `ceil(min_samples_split*n_samples)`

    min_samples_leaf : int/float (default=1)
        Defines requirement for a leaf node. The minimum number of samples needed to be a leaf node:

        - int, specifies the minimum number of samples
        - float, then represents a percentage. Minimum number of samples is computed as \
          `ceil(min_samples_split*n_samples)

    min_weight_fraction_leaf : float (default=0.0)
        Defines requirement for a leaf node. The minimum weight percentage of the sum total of the weights of \
        all input samples.
    max_features : int, float, string or None (default=None)
        Defines number of features to consider for the best possible split:

        - None, all specified features are used (oracle.feature_names)
        - int, uses specified values as `max_features` at each split.
        - float, as a percentage. Value for split is computed as `int(max_features * n_features)`.
        - "auto", `max_features=sqrt(n_features)`.
        - "sqrt", `max_features=sqrt(n_features)`.
        - "log2", `max_features=log2(n_features)`.

    seed : int, (default=None)
        seed for random number generator
    max_leaf_nodes : int or None (default=None)
        TreeSurrogates are constructed top-down in best first manner(best decrease in relative impurity).
        If None, results in maximum possible number of leaf nodes. This tends to over-fitting.
    min_impurity_decrease : float (default=0.0)
        Tree node is considered for splitting if relative decrease in impurity is >= `min_impurity_decrease`.
    class_weight : dict, list of dicts, str ("balanced" or None) (default="balanced")
        Weights associated with classes for handling data imbalance:

        - None, all classes have equal weights
        - "balanced", adjusts the class weights automatically. Weights are assigned inversely proportional \
          to class frequencies ``n_samples / (n_classes * np.bincount(y))``

    presort : bool (default=False)
        Sorts the data before building surrogates trees to find the best splits. When dealing with larger datasets, \
        setting it to True might result in increasing computation time because of the pre sorting operation.
    impurity_threshold : float (default=0.01)
        Specifies the acceptable disparity between the Oracle and TreeSurrogates. The higher the difference between \
        the Oracle and TreeSurrogate less faithful are the explanations generated.

    Attributes
    ----------
    oracle : skater.model.local_model.InMemoryModel
        The fitted base model with the prediction function
    feature_names: list of str
        Names of the features considered.
    estimator_ : DecisionTreeClassifier/DecisionTreeRegressor
        The Surrogate estimator.
    estimator_type_ : str
        Surrogate estimator type ("classifier" or "regressor").
    best_score_ : numpy.float64
        Surrogate estimator's best score post pre-pruning.
    scorer_name_ : str
        Scorer used for optimizing the surrogate estimator

    Examples
    --------

    >>> interpreter = Interpretation(X_train, feature_names=iris.feature_names)
    >>> model_inst = InMemoryModel(clf.predict, examples=X_train, model_type='classifier', unique_values=[0, 1],
    >>>                       feature_names=iris.feature_names, target_names=iris.target_names, log_level=_INFO)
    >>> # Using the interpreter instance invoke call to the TreeSurrogate
    >>> surrogate_explainer = interpreter.tree_surrogate(oracle=model_inst, seed=5)
    >>> surrogate_explainer.fit(X_train, y_train, use_oracle=True, prune='post', scorer_type='default')
    >>> surrogate_explainer.plot_global_decisions(colors=['coral', 'lightsteelblue','darkkhaki'],
    >>>                                          file_name='simple_tree_pre.png')
    >>> show_in_notebook('simple_tree_pre.png', width=400, height=300)

    References
    ----------
    .. [1] Mark W. Craven(1996) EXTRACTING COMPREHENSIBLE MODELS FROM TRAINED NEURAL NETWORKS
           (http://ftp.cs.wisc.edu/machine-learning/shavlik-group/craven.thesis.pdf)
    .. [2] Mark W. Craven and Jude W. Shavlik(NIPS, 96). Extracting Thee-Structured Representations of Thained Networks
           (https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)
    .. [3] DecisionTreeClassifier: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    .. [4] DecisionTreeRegressor: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """
    __name__ = "TreeSurrogate"

    def __init__(self, oracle=None, splitter='best', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, seed=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None, class_weight="balanced",
                 presort=False, impurity_threshold=0.01):

        if not isinstance(oracle, ModelType):
            raise ModelError("Incorrect estimator used, create one with skater.model.local.InMemoryModel")
        self.oracle = oracle
        self.logger = build_logger(oracle.logger.level, __name__)
        self.__model_type = None
        self.feature_names = oracle.feature_names
        self.class_names = oracle.target_names
        self.impurity_threshold = impurity_threshold
        self.criterion_types = {'classifier': {'criterion': ['gini', 'entropy']},
                                'regressor': {'criterion': ['mse', 'friedman_mse', 'mae']}}
        self.splitter_types = ['best', 'random']
        self.splitter = splitter if any(splitter in item for item in self.splitter_types) else 'best'
        self.seed = seed
        self.__model_type = oracle.model_type
        self.__scorer_name = None
        self.__best_score = None

        # TODO validate the parameters based on estimator type
        if self.__model_type == 'classifier':
            est = DecisionTreeClassifier(splitter=self.splitter, max_depth=max_depth,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                         max_features=max_features, random_state=seed,
                                         max_leaf_nodes=max_leaf_nodes,
                                         min_impurity_decrease=min_impurity_decrease,
                                         class_weight=class_weight, presort=presort)
        elif self.__model_type == 'regressor':
            est = DecisionTreeRegressor(splitter=self.splitter, max_depth=None,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        max_features=max_features,
                                        random_state=seed, max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_split=min_impurity_split, presort=presort)
        else:
            raise ModelError("Model type not supported. Supported options types{'classifier', 'regressor'}")
        self.__model = est
        self.__pred_func = lambda X, prob: self.__model.predict(X) if prob is False else self.__model.predict_proba(X)


    @staticmethod
    def __optimizer_condition(o_s, new_s, scoring_type, threshold):
        # if optimizing on a loss function then the type is decreasing
        # vs optimizing on a model metric which is increasing
        if scoring_type == 'decreasing':
            return round(o_s, 3) + threshold >= round(new_s, 3)
        else:
            return round(o_s, 3) - threshold <= round(new_s, 3)


    def _post_pruning(self, X, Y, scorer_type, impurity_threshold, needs_prob=False):
        self.__model.fit(X, Y)
        y_pred = self.__pred_func(X, needs_prob)
        # makes sense for classification use-case, be cautious when enabling for regression
        self.logger.debug("Unique Labels in ground truth provided {}".format(np.unique(Y)))
        if needs_prob is False:
            self.logger.debug("Unique Labels in predictions generated {}".format(np.unique(y_pred)))
        else:
            self.logger.debug("Probability scoring is enabled min:{}/max:{}".format(np.min(y_pred), np.max(y_pred)))

        scorer = self.oracle.scorers.get_scorer_function(scorer_type=scorer_type)
        self.logger.info("Scorer used {}".format(scorer.name))
        original_score = scorer(Y, y_pred)
        self.logger.info("original score using base model {}".format(original_score))

        tree = self.__model.tree_
        no_of_nodes = tree.node_count
        tree_leaf = -1  # value to identify a leaf node in a tree

        removed_node_index = []
        for index in range(no_of_nodes):
            current_left, current_right = tree.children_left[index], tree.children_right[index]
            if tree.children_left[index] != tree_leaf or tree.children_right[index] != tree_leaf:
                tree.children_left[index], tree.children_right[index] = -1, -1
                new_score = scorer(Y, self.__pred_func(X, needs_prob))
                self.logger.debug("new score generate {}".format(new_score))

                if TreeSurrogate.__optimizer_condition(original_score, new_score, scorer.type, impurity_threshold):
                    removed_node_index.append(index)
                    self.logger.debug("Removed nodes: (index:{}-->[left node: {}, right node: {}])"
                                      .format(index, current_left, current_right))
                else:
                    tree.children_left[index], tree.children_right[index] = current_left, current_right
                    self.logger.debug("Added index {} back".format(index))
        self.logger.info("Summary: childrens of the following nodes are removed {}".format(removed_node_index))


    def _pre_pruning(self, X, Y, scorer_type, cv=5, n_iter_search=10, n_jobs=1, param_grid=None, verbose=False):
        default_grid = {
            "criterion": self.criterion_types[self.__model_type]['criterion'],
            "max_depth": [2, 4, 6, 8, 10, 12],  # helps in reducing the depth of the tree
            "min_samples_leaf": [2, 4],  # restrict the minimum number of samples in a leaf
            "max_leaf_nodes": [2, 4, 6, 8, 10]  # reduce the number of leaf nodes
        }
        search_space = param_grid if param_grid is not None else default_grid
        self.logger.debug("Default search space used for CV : {}".format(search_space))
        # Cost function aiming to optimize(Total Cost) = measure of fit + measure of complexity
        # References for pruning:
        # 1. http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # 2. https://www.coursera.org/lecture/ml-classification/optional-pruning-decision-trees-to-avoid-overfitting-qvf6v
        # Using Randomize Search here to prune the trees to improve readability without
        # comprising on model's performance
        scorer = self.oracle.scorers.get_scorer_function(scorer_type=scorer_type)
        self.logger.info("Scorer used {}".format(scorer.name))
        scorering_func = make_scorer(scorer, greater_is_better=scorer.type)
        verbose_level = 0 if verbose is False else 4
        random_search_estimator = RandomizedSearchCV(estimator=self.__model, cv=cv, param_distributions=search_space,
                                                     scoring=scorering_func, n_iter=n_iter_search, n_jobs=n_jobs,
                                                     random_state=self.seed, verbose=verbose_level)
        # train a surrogate DT
        random_search_estimator.fit(X, Y)
        # access the best estimator
        self.__model = random_search_estimator.best_estimator_
        self.__best_score = random_search_estimator.best_score_


    def fit(self, X, Y, use_oracle=True, prune='post', cv=5, n_iter_search=10,
            scorer_type='default', n_jobs=1, param_grid=None, impurity_threshold=0.01, verbose=False):
        """ Learn an approximate representation by constructing a Decision Tree based on the results retrieved by
        querying the Oracle(base model). Instances used for training should belong to the base learners instance space.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame
            Training input samples
        Y : numpy.ndarray, target values(ground truth)
        use_oracle : bool (defaul=True)
            Use of Oracle, helps the Surrogate model train on the decision boundaries learned by the base model. \
            The closer the surrogate model is to the Oracle, more faithful are the explanations.

              - True, builds a surrogate model against the predictions of the base model(Oracle).
              - False, learns an interpretable tree based model using the supplied training examples and ground truth.

        prune : None, str (default="post")
            Pruning is a useful technique to control the complexity of the tree (keeping the trees comprehensive \
            and interpretable) without compromising on model's accuracy. Avoiding to build large and deep trees \
            also helps in preventing over-fitting.

              - "pre"
              Also known as forward/online pruning. This pruning process uses a termination \
              condition(high and low thresholds) to prematurely terminate some of the branches and nodes.
              Cross Validation is applied to measure the goodness of the fit while the tree is pruned.

              - "pos"
              Also known as backward pruning. The pruning process is applied post the construction of the \
              tree using the specified model parameters. This involves reducing the branches and nodes using \
              a cost function. The current implementation support cost optimization using \
              Model's scoring metrics(e.g. r2, log-loss, f1, ...).

        cv : int, (default=5)
            Randomized cross validation used only for 'pre-pruning' right now.
        n_iter_search : int (default=10)
            Number of parameter setting combinations that are sampled for pre-pruning.
        scorer_type : str (default="default")
        n_jobs : int (default=1)
            Number of jobs to run in parallel.
        param_grid : dict
            Dictionary of parameters to specify the termination condition for pre-pruning.
        impurity_threshold : float (default=0.01)
            Specifies acceptable performance drop when using Tree based surrogates to replicate the decision policies
            learned by the Oracle
        verbose : bool (default=False)
            Helps control the verbosity.

        References
        ----------
        .. [1] Nikita Patel and Saurabh Upadhyay(2012)
               Study of Various Decision Tree Pruning Methods with their Empirical Comparison in WEKA
               (https://pdfs.semanticscholar.org/025b/8c109c38dc115024e97eb0ede5ea873fffdb.pdf)
        """

        if verbose:
            self.logger.setLevel(_DEBUG)
        else:
            self.logger.setLevel(_INFO)
        # DataManager does type checking as well
        dm = DataManager(X, Y)
        X, Y = dm.X, dm.y
        # Below is an anti-pattern but had to use it. Should fix it in the long term
        y_hat_original = self.oracle._execute(X)

        # TODO: Revisit the check on using probability or class labels
        if use_oracle and self.oracle.probability:
            y_train = np.array(list(map(np.argmax, y_hat_original)))
        elif use_oracle:
            y_train = y_hat_original
        else:
            # this is when y_train is being passed and the desire is to build an interpretable tree based model
            y_train = Y

        if prune is None:
            self.logger.info("No pruning applied ...")
            self.__model.fit(X, y_train)
        elif prune == 'pre':
            # apply randomized cross validation for pruning
            self.logger.info("pre pruning applied ...")
            self._pre_pruning(X, y_train, scorer_type, cv, n_iter_search, n_jobs, param_grid, verbose)
        else:
            self.logger.info("post pruning applied ...")
            # Since, this is post pruning, we first learn a model
            # and then try to prune the tree controling the model's score using the impurity_threshold
            self._post_pruning(X, y_train, scorer_type, impurity_threshold, needs_prob=self.oracle.probability)
        y_hat_surrogate = self.__pred_func(X, self.oracle.probability)
        self.logger.info('Done generating prediction using the surrogate, shape {}'.format(y_hat_surrogate.shape))

        # Default metrics:
        # {Classification: if probability score used --> cross entropy(log-loss) else --> F1 score}
        # {Regression: Mean Absolute Error (MAE)}
        scorer = self.oracle.scorers.get_scorer_function(scorer_type=scorer_type)
        self.__scorer_name = scorer.name

        oracle_score = round(scorer(Y, y_hat_original), 3)
        # Since surrogate model is build against the base model's(Oracle's) predicted
        # behavior y_true=y_train
        surrogate_score = round(scorer(y_train, y_hat_surrogate), 3)
        self.logger.info('Done scoring, surrogate score {}; oracle score {}'.format(surrogate_score, oracle_score))

        impurity_score = round(oracle_score - surrogate_score, 3)
        if impurity_score > self.impurity_threshold:
            self.logger.warning('impurity score: {} of the surrogate model is higher than the impurity threshold: {}. '
                                'The higher the impurity score, lower is the fidelity/faithfulness '
                                'of the surrogate model'.format(impurity_score, impurity_threshold))
        return impurity_score


    @property
    def estimator_(self):
        """ Learned approximate surrogate estimator
        """
        return self.__model


    @property
    def estimator_type_(self):
        """ Estimator type
        """
        return self.__model_type


    @property
    def best_score_(self):
        """ Best score post pre-pruning
        """
        return self.__best_score


    @property
    def scorer_name_(self):
        """ Cost function used for optimization
        """
        return self.__scorer_name


    def predict(self, X, prob_score=False):
        """ Predict for input X
        """
        predict_values = self.__model.predict(X)
        predict_prob_values = self.__model.predict_proba(X) if prob_score is True else None
        return predict_values if predict_prob_values is None else predict_prob_values


    def plot_global_decisions(self, colors=None, enable_node_id=True, random_state=0, file_name="interpretable_tree.png",
                              show_img=False, fig_size=(20, 8)):
        """ Visualizes the decision policies of the surrogate tree.
        """
        graph_inst = plot_tree(self.__model, self.__model_type, feature_names=self.feature_names, color_list=colors,
                               class_names=self.class_names, enable_node_id=enable_node_id, seed=random_state)
        f_name = "interpretable_tree.png" if file_name is None else file_name
        graph_inst.write_png(f_name)

        try:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
        except ImportError:
            raise MatplotlibUnavailableError("Matplotlib is required but unavailable on the system.")
        except RuntimeError:
            raise MatplotlibDisplayError("Matplotlib unable to open display")

        if show_img:
            plt.rcParams["figure.figsize"] = fig_size
            img = plt.imread(f_name)
            if self.__model_type == 'regressor':
                cax = plt.imshow(img, cmap=plt.cm.get_cmap(graph_inst.get_colorscheme()))
                plt.colorbar(cax)
            else:
                plt.imshow(img)
        return graph_inst


    def decisions_as_txt(self, scope='global', X=None):
        """ Retrieve the decision policies as text
        """
        tree_to_text(self.__model, self.feature_names, self.__model_type, scope, X)



"""Model subclass for in memory predict functions"""



class InMemoryModel(ModelType):
    """
    This model can be called directly from memory
    """

    def __init__(self,
                 prediction_fn,
                 input_formatter=None,
                 output_formatter=None,
                 target_names=None,
                 feature_names=None,
                 unique_values=None,
                 examples=None,
                 model_type=None,
                 probability=None,
                 log_level=30):
        """This model can be called directly from memory

        Parameters
        ----------
        prediction_fn: callable
            function that returns predictions

        input_formatter: callable
            This function will run on input data before passing
            to the prediction_fn. This usually should take your data type
            and convert them to numpy arrays or dataframes.

        output_formatter: callable
            This function will run on input data before passing
            to the prediction_fn. This usually should take your data type
            and convert them to numpy arrays or dataframes.

        target_names: array type
            (optional) names of classes that describe model outputs.

        feature_names: array type
            (optional) Names of features the model consumes.

        unique_values: array type
            The set of possible output values. Only use on classifier models that
            return "best guess" predictions, not probability scores, e.g.

            model.predict(fruit1) -> 'apple'
            model.predict(fruit2) -> 'banana'

            ['apple','banana'] are the unique_values of the classifier

        examples: numpy.array or pandas.dataframe
            optional examples to use to make inferences about the function.

        model_type: None, "classifier", "regressor"
            Indicates which type of model is being used. If left as None, will try to infer based on the
            signature of the output type.

        probability: None, True, False
            If using a classifier, indicates whether probabilities are provided
            (as opposed to indicators/labels).

        log_level: int
            config setting to see model logs. 10 is a good value for seeing debug messages.
            30 is warnings only.


        """

        if not hasattr(prediction_fn, "__call__"):
            raise(ModelError("Predict function must be callable"))

        self.prediction_fn = prediction_fn
        super(InMemoryModel, self).__init__(log_level=log_level,
                                            target_names=target_names,
                                            examples=examples,
                                            unique_values=unique_values,
                                            input_formatter=input_formatter,
                                            output_formatter=output_formatter,
                                            feature_names=feature_names,
                                            model_type=model_type,
                                            probability=probability,
                                            )


    def _execute(self, *args, **kwargs):
        """
        Just use the function itself for predictions
        """
        return self.prediction_fn(*args, **kwargs)


    @staticmethod
    def _predict(data, predict_fn, input_formatter, output_formatter, transformer):
        """Static prediction function for multiprocessing usecases

        Parameters
        ----------
        data: arraytype

        formatter: callable
            function responsible for formatting model outputs as necessary. For instance,
            one hot encoding multiclass outputs.

        predict_fn: callable

        Returns
        -----------
        predictions: arraytype
        """
        results = output_formatter(predict_fn(input_formatter(data)))
        if transformer:
            return transformer(results)
        else:
            return results


    def _get_static_predictor(self):

        predict_fn = partial(self._predict,
                             transformer=self.transformer,
                             predict_fn=self.prediction_fn,
                             input_formatter=self.input_formatter,
                             output_formatter=self.output_formatter,
                             )
        return predict_fn



"""Interpretation Class"""



class Interpretation(object):
    """
    Interpretation class. Before calling interpretation subclasses like partial
    dependence, one must call Interpretation.load_data().


    Examples
    --------
        >>> interpreter = Interpretation()
        >>> interpreter.load_data(X, feature_ids = ['a','b'])
        >>> interpreter.partial_dependence([feature_id1, feature_id2], regressor.predict)
    """

    def __init__(self, training_data=None, training_labels=None, class_names=None, feature_names=None, index=None,
                 log_level=30):
        """
        Attaches local and global interpretations
        to Interpretation object.

        Parameters
        -----------
        log_level: int
            Logger Verbosity, see https://docs.python.org/2/library/logging.html
            for details.

        """
        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)
        self.data_set = None
        self.feature_names = feature_names
        self.class_names = class_names
        self.load_data(training_data,
                       training_labels=training_labels,
                       feature_names=feature_names,
                       index=index)
        self.tree_surrogate = TreeSurrogate


    def load_data(self, training_data, training_labels=None, feature_names=None, index=None):
        """
        Creates a DataSet object from inputs, ties to interpretation object.
        This will be exposed to all submodules.

        Parameters
        ----------
        training_data: numpy.ndarray, pandas.DataFrame
            the dataset. can be 1D or 2D

        feature_names: array-type
            names to call features.

        index: array-type
            names to call rows.


        Returns
        --------
            None
        """

        self.logger.info("Loading Data")
        self.data_set = DataManager(training_data,
                                    y=training_labels,
                                    feature_names=feature_names,
                                    index=index,
                                    log_level=self._log_level)
        self.logger.info("Data loaded")
        self.logger.info("Data shape: {}".format(self.data_set.X.shape))
        self.logger.info("Dataset Feature_ids: {}".format(self.data_set.feature_ids))
