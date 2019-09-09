# file: pivoted_qr_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-PIVOTED_QR_BATCH"></a>
## \example pivoted_qr_dense_batch.py

import os
import sys

from daal.algorithms import pivoted_qr
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
dataFileName = os.path.join(DAAL_PREFIX, 'batch', 'qr.csv')

if __name__ == "__main__":

    # Initialize FileDataSource to retrieve input data from .csv file
    dataSource = FileDataSource(
        dataFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from input file
    dataSource.loadDataBlock()

    # Create algorithm objects for pivoted_qr decomposition computing in batch mode
    algorithm = pivoted_qr.Batch()

    # Set input arguments of the algorithm
    algorithm.input.set(pivoted_qr.data, dataSource.getNumericTable())

    # Get computed pivoted_qr decomposition
    res = algorithm.compute()

    # Print values
    printNumericTable(res.get(pivoted_qr.matrixQ), "Orthogonal matrix Q:", 10)
    printNumericTable(res.get(pivoted_qr.matrixR), "Triangular matrix R:")
    printNumericTable(res.get(pivoted_qr.permutationMatrix), "Permutation matrix P:")
