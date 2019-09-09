/* file: pca_dense_svd_batch_fpt_cpu.cpp */
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

//++
//  Implementation of PCA calculation functions.
//--

#include "pca_dense_svd_batch_container.h"
#include "pca_dense_svd_batch_kernel_instance.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
template class BatchContainer<DAAL_FPTYPE, svdDense, DAAL_CPU>;
}
}
}
}
