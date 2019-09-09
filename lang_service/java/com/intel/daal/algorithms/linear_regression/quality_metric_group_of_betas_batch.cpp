/* file: quality_metric_group_of_betas_batch.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
#include "daal.h"
#include "com_intel_daal_algorithms_linear_regression_quality_metric_GroupOfBetasBatch.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression::quality_metric;
using namespace daal::algorithms::linear_regression::quality_metric::group_of_betas;

/*
* Class:     com_intel_daal_algorithms_linear_regression_quality_metric_GroupOfBetasBatch
* Method:    cInit
* Signature: (IIJJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_quality_1metric_GroupOfBetasBatch_cInit
(JNIEnv *, jobject, jint prec, jint method, jlong nBeta, jlong nBetaReducedModel)
{
    return jniBatch<group_of_betas::Method, Batch, defaultDense>::newObj(prec, method, nBeta, nBetaReducedModel);
}

/*
* Class:     com_intel_daal_algorithms_linear_regression_quality_metric_GroupOfBetasBatch
* Method:    cSetResult
* Signature: (JIIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_quality_1metric_GroupOfBetasBatch_cSetResult
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<group_of_betas::Method, Batch, defaultDense>::
        setResult<group_of_betas::Result>(prec, method, algAddr, resultAddr);
}


/*
* Class:     com_intel_daal_algorithms_linear_regression_quality_metric_GroupOfBetasBatch
* Method:    cInitParameter
* Signature: (JIIJJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_quality_1metric_GroupOfBetasBatch_cInitParameter
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method, jlong nBeta, jlong nBetaReducedModel)
{
    group_of_betas::Parameter *parameterAddr = (group_of_betas::Parameter *)jniBatch<group_of_betas::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
    if(parameterAddr)
    {
        parameterAddr->numBeta = nBeta;
        parameterAddr->numBetaReducedModel = nBetaReducedModel;
    }

    return (jlong)parameterAddr;
}

/*
* Class:     com_intel_daal_algorithms_linear_regression_quality_metric_GroupOfBetasBatch
* Method:    cGetInput
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_quality_1metric_GroupOfBetasBatch_cGetInput
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<group_of_betas::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_linear_regression_quality_metric_GroupOfBetasBatch
* Method:    cGetResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_quality_1metric_GroupOfBetasBatch_cGetResult
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<group_of_betas::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_linear_regression_quality_metric_GroupOfBetasBatch
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_quality_1metric_GroupOfBetasBatch_cClone
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<group_of_betas::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
