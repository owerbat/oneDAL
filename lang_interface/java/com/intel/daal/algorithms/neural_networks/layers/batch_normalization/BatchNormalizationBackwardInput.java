/* file: BatchNormalizationBackwardInput.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @defgroup batch_normalization_backward Backward Batch Normalization Layer
 * @brief Contains classes for the backward batch normalization layer
 * @ingroup batch_normalization
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONBACKWARDINPUT"></a>
 * @brief Input object for the backward batch normalization layer
 */
public final class BatchNormalizationBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public BatchNormalizationBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward batch normalization layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(BatchNormalizationLayerDataId id, Tensor val) {
        if (id == BatchNormalizationLayerDataId.auxData || id == BatchNormalizationLayerDataId.auxWeights ||
            id == BatchNormalizationLayerDataId.auxMean || id == BatchNormalizationLayerDataId.auxStandardDeviation ||
            id == BatchNormalizationLayerDataId.auxPopulationMean || id == BatchNormalizationLayerDataId.auxPopulationVariance) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect BatchNormalizationLayerDataId");
        }
    }

    /**
     * Returns the input object of the backward batch normalization layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(BatchNormalizationLayerDataId id) {
        if (id == BatchNormalizationLayerDataId.auxData || id == BatchNormalizationLayerDataId.auxWeights ||
            id == BatchNormalizationLayerDataId.auxMean || id == BatchNormalizationLayerDataId.auxStandardDeviation ||
            id == BatchNormalizationLayerDataId.auxPopulationMean || id == BatchNormalizationLayerDataId.auxPopulationVariance) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
