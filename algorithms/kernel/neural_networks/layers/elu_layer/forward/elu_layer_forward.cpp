/* file: elu_layer_forward.cpp */
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

/*
//++
//  Implementation of ELU calculation algorithm and types methods.
//--
*/

#include "elu_layer_forward_types.h"
#include "elu_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELU_FORWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns dimensions of weights tensor
 * \return Dimensions of weights tensor
 */
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

/** \brief Default constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Returns the result of the forward ELU layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::TensorPtr Result::get(LayerDataId id) const
{
    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    if (!layerData) { return data_management::TensorPtr(); }
    return data_management::Tensor::cast((*layerData)[id]);
}

/**
 * Sets the result of the forward ELU layer
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(LayerDataId id, const data_management::TensorPtr &value)
{
    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    if (layerData) { (*layerData)[id] = value; }
}

/**
 * Checks the result of the forward ELU layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input,
                               const daal::algorithms::Parameter *parameter,
                               int method) const
{
    services::Status status;

    DAAL_CHECK_STATUS(status, layers::forward::Result::check(input, parameter, method));

    auto in  = static_cast<const Input *>(input);
    auto par = static_cast<const layers::Parameter * >(parameter);
    auto &dataDimensions = in->get(layers::forward::data)->getDimensions();

    DAAL_CHECK_TENSOR(status, get(layers::forward::value).get(), valueStr(), &dataDimensions);

    if (!par->predictionStage)
    {
        DAAL_CHECK_TENSOR(status, get(elu::auxData).get(), auxDataStr(), &dataDimensions);
        DAAL_CHECK_TENSOR(status, get(elu::auxIntermediateValue).get(), auxIntermediateValueStr(), &dataDimensions);
    }

    return status;
}

/**
 * Sets the result that is used in backward ELU layer
 * \param[in] input     Pointer to an object containing the input data
 */
services::Status Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
    set(elu::auxData, in->get(layers::forward::data));
    return services::Status();
}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par,
                                                        const int method) const
{
    return inputSize;
}

}// namespace interface1
}// namespace forward
}// namespace elu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
