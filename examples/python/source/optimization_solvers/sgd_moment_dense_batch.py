# file: sgd_moment_dense_batch.py
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

#
# !  Content:
# !    Python example of the Stochastic gradient descent algorithm
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-SGD_MOMENT_DENSE_BATCH"></a>
##  \example sgd_moment_dense_batch.py
#

import os
import sys

import numpy as np

import daal.algorithms.optimization_solver as optimization_solver
import daal.algorithms.optimization_solver.mse
import daal.algorithms.optimization_solver.sgd
import daal.algorithms.optimization_solver.iterative_solver

from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'mse.csv')

nFeatures = 3
accuracyThreshold = 0.0000001
nIterations = 1000
batchSize = 4
learningRate = 0.5
initialPoint = np.array([[8], [2], [1], [4]], dtype=np.float64)

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileName,
                                DataSourceIface.notAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Create Numeric Tables for data and values for dependent variable
    data = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    dependentVariables = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(data, dependentVariables)

    # Retrieve the data from the input file
    dataSource.loadDataBlock(mergedData)

    nVectors = data.getNumberOfRows()

    mseObjectiveFunction = optimization_solver.mse.Batch(nVectors)
    mseObjectiveFunction.input.set(optimization_solver.mse.data, data)
    mseObjectiveFunction.input.set(optimization_solver.mse.dependentVariables, dependentVariables)

    # Create objects to compute the Stochastic gradient descent result using the momentum method
    sgdMomentumAlgorithm = optimization_solver.sgd.Batch(mseObjectiveFunction, method=optimization_solver.sgd.momentum)

    # Set input objects for the the Stochastic gradient descent algorithm
    sgdMomentumAlgorithm.input.setInput(optimization_solver.iterative_solver.inputArgument,
                                        HomogenNumericTable(initialPoint))
    sgdMomentumAlgorithm.parameter.learningRateSequence = HomogenNumericTable(1, 1, NumericTableIface.doAllocate,
                                                                              learningRate)
    sgdMomentumAlgorithm.parameter.nIterations = nIterations
    sgdMomentumAlgorithm.parameter.batchSize = batchSize
    sgdMomentumAlgorithm.parameter.accuracyThreshold = accuracyThreshold

    # Compute the Stochastic gradient descent result
    # Result class from daal.algorithms.optimization_solver.iterative_solver
    res = sgdMomentumAlgorithm.compute()

    # Print computed the Stochastic gradient descent result
    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Minimum")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")