import pandas as pd
import numpy as np
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator  # for single tree
from h2o.backend import H2OLocalServer
import subprocess
from IPython.display import Image
from IPython.display import display


class TREPAN(object):
    def __init__(self, model, X_test, feature_names, max_depth, model_id='surrogate_mojo'):
        self.model = model
        self.X_test = np.array(X_test)
        self.y_test = np.array(model.predict(X_test))
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.model_id = model_id
        h2o.init(max_mem_size='2G')  # start h2o
        h2o.remove_all()  # remove any existing data structures from h2o memory

    def display_tree(self, title, shell=False):
        if not shell:
            predict_probs = pd.DataFrame(self.model.predict_proba(self.X_test)[:, 1], columns=['prob_pred'])
            test = pd.DataFrame(np.c_[self.X_test, self.y_test], columns=self.feature_names.append(pd.Index(['label'])))
            test_yhat = h2o.H2OFrame(pd.concat([test, predict_probs], axis=1))

            # initialize single tree surrogate model
            surrogate = H2ORandomForestEstimator(ntrees=1,  # use only one tree
                                                 sample_rate=1,  # use all rows in that tree
                                                 mtries=-2,  # use all columns in that tree
                                                 max_depth=self.max_depth,  # shallow trees are easier to understand
                                                 seed=12345,  # random seed for reproducibility
                                                 model_id=self.model_id)  # gives MOJO artifact a recognizable name

            # train single tree surrogate model
            surrogate.train(x=list(self.feature_names), y='label', training_frame=test_yhat)

            # persist MOJO (compiled, representation of trained model)
            # from which to generate plot of surrogate
            mojo_path = surrogate.download_mojo(path='.')

            # locate h2o jar
            hs = H2OLocalServer()
            h2o_jar_path = hs._find_jar()

            # construct command line call to generate graphviz version of 
            # surrogate tree see for more information: 
            # http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html
            model_id = self.model_id
            gv_file_name = model_id + '.gv'
            gv_args = str('-cp ' + h2o_jar_path +
                          ' hex.genmodel.tools.PrintMojo --tree 0 -i '
                          + mojo_path + ' -o').split()
            gv_args.insert(0, 'java')
            gv_args.append(gv_file_name)
            if title is not None:
                gv_args = gv_args + ['--title', title]

            # if the line below is failing for you, try instead:
            # _ = subprocess.call(gv_args, shell=True)  
            _ = subprocess.call(gv_args)

            # construct call to generate PNG from 
            # graphviz representation of the tree
            png_file_name = model_id + '.png'
            png_args = str('dot -Tpng ' + gv_file_name + ' -o ' + png_file_name)
            png_args = png_args.split()

            # if the line below is failing for you, try instead:
            # _ = subprocess.call(png_args, shell=True)  
            _ = subprocess.call(png_args)

            # display in-notebook
            display(Image((png_file_name)))
