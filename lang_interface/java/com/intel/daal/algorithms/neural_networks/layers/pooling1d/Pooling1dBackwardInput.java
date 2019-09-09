/* file: Pooling1dBackwardInput.java */
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
 * @defgroup pooling1d_backward Backward One-dimensional Pooling Layer
 * @brief Contains classes for backward one-dimensional (1D) pooling layer
 * @ingroup pooling1d
 * @{
 */
/**
 * @brief Contains classes of the one-dimensional (1D) pooling layers
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling1d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DBACKWARDINPUT"></a>
 * @brief Input object for the backward one-dimensional pooling layer
 */
public class Pooling1dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Pooling1dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
