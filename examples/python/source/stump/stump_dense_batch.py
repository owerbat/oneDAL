# file: stump_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-STUMP_BATCH"></a>
## \example stump_dense_batch.py

import os
import sys

from daal.algorithms import classifier
from daal.algorithms.stump import training, prediction
from daal.data_management import (
    FileDataSource, DataSourceIface, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables

DAAL_PREFIX = os.path.join('..', 'data')

#  Input data set parameters
nFeatures = 20
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'stump_train.csv')
testDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'stump_test.csv')

trainingResult = None
predictionResult = None
testGroundTruth = None


def trainModel():
    global trainingResult
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName,
        DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)

    # Retrieve the data from the input file
    trainDataSource.loadDataBlock(mergedData)

    #  Create an algorithm object to train the stump model
    algorithm = training.Batch()

    #  Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)

    #  Compute and retrieve the algorithm results
    trainingResult = algorithm.compute()


def testModel():
    global predictionResult, testGroundTruth

    #  Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName,
        DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    testGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Retrieve the data from the input file
    testDataSource.loadDataBlock(mergedData)

    #  Create an algorithm object to train the stump model
    algorithm = prediction.Batch()

    #  Pass a training data set and dependent values to the algorithm
    algorithm.input.setTable(classifier.prediction.data, testData)
    algorithm.input.setModel(classifier.prediction.model,
                             trainingResult.get(classifier.training.model))

    #  Compute and retrieve the algorithm Result class from classifier.prediction
    predictionResult = algorithm.compute()


def printResults():
    printNumericTables(
        testGroundTruth,
        predictionResult.get(classifier.prediction.prediction),
        "Ground truth", "Classification results",
        "Stump classification results (first 20 observations):", 20, flt64=False)

if __name__ == "__main__":
    trainModel()
    testModel()
    printResults()
