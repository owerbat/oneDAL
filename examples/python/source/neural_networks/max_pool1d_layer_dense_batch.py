# file: max_pool1d_layer_dense_batch.py
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
# !    Python example of neural network forward and backward one-dimensional maximum pooling layers usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-MAXIMUM_POOLING1D_LAYER_BATCH"></a>
## \example max_pool1d_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV

# Input data set name
datasetFileName = os.path.join("..", "data", "batch", "layer.csv")

if __name__ == "__main__":

    # Read datasetFileName from a file and create a tensor to store input data
    data = readTensorFromCSV(datasetFileName)
    nDim = data.getNumberOfDimensions()

    printTensor(data, "Forward one-dimensional maximum pooling layer input (first 10 rows):", 10)

    # Create an algorithm to compute forward one-dimensional pooling layer results using maximum method
    forwardLayer = layers.maximum_pooling1d.forward.Batch(nDim)
    forwardLayer.input.setInput(layers.forward.data, data)

    # Compute forward one-dimensional maximum pooling layer results
    forwardResult = forwardLayer.compute()

    # Print the results of the forward one-dimensional maximum pooling layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward one-dimensional maximum pooling layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(layers.maximum_pooling1d.auxSelectedIndices),
                "Forward one-dimensional maximum pooling layer selected indices (first 5 rows):", 5)

    # Create an algorithm to compute backward one-dimensional maximum pooling layer results using default method
    backwardLayer = layers.maximum_pooling1d.backward.Batch(nDim)

    # Set input objects for the backward one-dimensional maximum pooling layer
    backwardLayer.input.setInput(layers.backward.inputGradient, forwardResult.getResult(layers.forward.value))
    backwardLayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward one-dimensional maximum pooling layer results
    backwardResult = backwardLayer.compute()

    # Print the results of the backward one-dimensional maximum pooling layer
    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Backward one-dimensional maximum pooling layer result (first 10 rows):", 10)
