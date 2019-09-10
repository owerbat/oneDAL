# file: qr_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-QR_BATCH"></a>
## \example qr_dense_batch.py

import os
import sys

from daal.algorithms import qr
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'qr.csv')
nRows = 16000

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve input data from .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from input file
    dataSource.loadDataBlock(nRows)

    # Create algorithm to compute QR decomposition
    algorithm = qr.Batch()

    algorithm.input.set(qr.data, dataSource.getNumericTable())

    res = algorithm.compute()

    # Print results
    printNumericTable(res.get(qr.matrixQ), "Orthogonal matrix Q:", 10)
    printNumericTable(res.get(qr.matrixR), "Triangular matrix R:")