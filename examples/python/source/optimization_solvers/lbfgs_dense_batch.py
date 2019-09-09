# file: lbfgs_dense_batch.py
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

# !  Content:
# !    Python example of the limited memory Broyden-Fletcher-Goldfarb-Shanno
# !    algorithm
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LBFGS_BATCH"></a>
##  \example lbfgs_dense_batch.py
#

import os
import sys

import numpy as np

import daal.algorithms.optimization_solver as optimization_solver
import daal.algorithms.optimization_solver.mse
import daal.algorithms.optimization_solver.lbfgs
import daal.algorithms.optimization_solver.iterative_solver

from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'lbfgs.csv')

nFeatures = 10
nIterations = 1000
stepLength = 1.0e-4

initialPoint = np.array([[100], [100], [100], [100], [100], [100], [100], [100], [100], [100], [100]], dtype=np.float64)
expectedPoint = np.array([[11], [  1], [  2], [  3], [  4], [  5], [  6], [  7], [  8], [  9], [ 10]], dtype=np.float64)

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileName,
                                DataSourceIface.notAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Create Numeric Tables for input data and dependent variables
    data = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    dependentVariables = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(data, dependentVariables)

    # Retrieve the data from input file
    dataSource.loadDataBlock(mergedData)

    mseObjectiveFunction = optimization_solver.mse.Batch(data.getNumberOfRows())
    mseObjectiveFunction.input.set(optimization_solver.mse.data, data)
    mseObjectiveFunction.input.set(optimization_solver.mse.dependentVariables, dependentVariables)

    # Create objects to compute LBFGS result using the default method
    algorithm = optimization_solver.lbfgs.Batch(mseObjectiveFunction)
    algorithm.parameter.nIterations = nIterations
    algorithm.parameter.stepLengthSequence = HomogenNumericTable(1, 1, NumericTableIface.doAllocate, stepLength)

    # Set input objects for LBFGS algorithm
    algorithm.input.setInput(optimization_solver.iterative_solver.inputArgument, HomogenNumericTable(initialPoint))

    # Compute LBFGS result
    # Result class from daal.algorithms.optimization_solver.iterative_solver
    res = algorithm.compute()

    expectedCoefficients = HomogenNumericTable(expectedPoint)

    # Print computed LBFGS results
    printNumericTable(expectedCoefficients, "Expected coefficients:")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.minimum), "Resulting coefficients:")
    printNumericTable(res.getResult(optimization_solver.iterative_solver.nIterations), "Number of iterations performed:")
