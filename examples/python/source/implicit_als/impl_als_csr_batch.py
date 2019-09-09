# file: impl_als_csr_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

## <a name="DAAL-EXAMPLE-PY-IMPLICIT_ALS_CSR_BATCH"></a>
## \example impl_als_csr_batch.py

import os
import sys

import daal.algorithms.implicit_als.prediction.ratings as ratings
import daal.algorithms.implicit_als.training as training
import daal.algorithms.implicit_als.training.init as init

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'implicit_als_csr.csv')

# Algorithm parameters
nFactors = 2

dataTable = None
initialModel = None
trainingResult = None


def initializeModel():
    global initialModel, dataTable

    # Read trainDatasetFileName from a file and create a numeric table to store the input data
    dataTable = createSparseTable(trainDatasetFileName)

    # Create an algorithm object to initialize the implicit ALS model with the default method
    initAlgorithm = init.Batch(method=init.fastCSR)
    initAlgorithm.parameter.nFactors = nFactors

    # Pass a training data set and dependent values to the algorithm
    initAlgorithm.input.set(init.data, dataTable)

    # Initialize the implicit ALS model
    res = initAlgorithm.compute()
    # (Result class from implicit_als.training.init)
    initialModel = res.get(init.model)


def trainModel():
    global trainingResult

    # Create an algorithm object to train the implicit ALS model with the default method
    algorithm = training.Batch(method=training.fastCSR)

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.setTable(training.data, dataTable)
    algorithm.input.setModel(training.inputModel, initialModel)

    algorithm.parameter.nFactors = nFactors

    # Build the implicit ALS model
    # Retrieve the algorithm results
    trainingResult = algorithm.compute()


def testModel():

    # Create an algorithm object to predict recommendations of the implicit ALS model
    algorithm = ratings.Batch()
    algorithm.parameter.nFactors = nFactors

    algorithm.input.set(ratings.model, trainingResult.get(training.model))

    res = algorithm.compute()

    predictedRatings = res.get(ratings.prediction)

    printNumericTable(predictedRatings, "Predicted ratings:")

if __name__ == "__main__":

    initializeModel()
    trainModel()
    testModel()
