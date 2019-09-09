/* file: model_builder.cpp */
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
#include "com_intel_daal_algorithms_svm_ModelBuilder.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::algorithms::svm;
using namespace daal::data_management;
using namespace daal::services;

/*
* Class:     com_intel_daal_algorithms_svm_ModelBuilder
* Method:    cInit
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cInit
(JNIEnv *, jobject, jint prec, jlong nFeatures, jlong nSupportVectors)
{
    if(prec == 0)
    {
        return (jlong)(new SharedPtr<ModelBuilder<double>>(new ModelBuilder<double>(nFeatures, nSupportVectors)));
    }
    else
    {
        return (jlong)(new SharedPtr<ModelBuilder<float>>(new ModelBuilder<float>(nFeatures, nSupportVectors)));
    }
}

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cGetModel
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cGetModel
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec)
{
    ModelPtr *model = new ModelPtr;
    if(prec == 0)
    {
        services::SharedPtr<ModelBuilder<double>> *ptr = new services::SharedPtr<ModelBuilder<double>>();
        *ptr = staticPointerCast<ModelBuilder<double>>(*(SharedPtr<ModelBuilder<double>> *)algAddr);
        *model = staticPointerCast<Model>((*ptr)->getModel());
    }
    else
    {
        services::SharedPtr<ModelBuilder<float>> *ptr = new services::SharedPtr<ModelBuilder<float>>();
        *ptr = staticPointerCast<ModelBuilder<float>>(*(SharedPtr<ModelBuilder<float>> *)algAddr);
        *model = staticPointerCast<Model>((*ptr)->getModel());
    }

    return (jlong)model;
}

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cSetBiasDouble
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cSetBiasDouble
(JNIEnv *env, jobject thisObj, jlong algAddr, jdouble bias)
{
    services::SharedPtr<ModelBuilder<double>> *ptr = new services::SharedPtr<ModelBuilder<double>>();
    *ptr = staticPointerCast<ModelBuilder<double>>(*(SharedPtr<ModelBuilder<double>> *)algAddr);
    (*ptr)->setBias(bias);
}

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cSetBiasFloat
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cSetBiasDouble
(JNIEnv *env, jobject thisObj, jlong algAddr, jfloat bias)
{
    services::SharedPtr<ModelBuilder<float>> *ptr = new services::SharedPtr<ModelBuilder<float>>();
    *ptr = staticPointerCast<ModelBuilder<float>>(*(SharedPtr<ModelBuilder<float>> *)algAddr);
    (*ptr)->setBias(bias);
}

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cSetSupportVectorsFloat
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cSetSupportVectorsFloat
(JNIEnv *env, jobject, jlong algAddr, jobject byteBuffer, jlong nValues)
{
    float *firstVal = (float *)(env->GetDirectBufferAddress(byteBuffer));
    float *lastVal = firstVal + nValues;
    services::SharedPtr<ModelBuilder<float>> *ptr = new services::SharedPtr<ModelBuilder<float>>();
    *ptr = staticPointerCast<ModelBuilder<float>>(*(SharedPtr<ModelBuilder<float>> *)algAddr);
    (*ptr)->setSupportVectors(firstVal, lastVal);
    DAAL_CHECK_THROW((*ptr)->getStatus());
}

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cSetSupportVectorsDouble
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cSetSupportVectorsDouble
(JNIEnv *env, jobject, jlong algAddr, jobject byteBuffer, jlong nValues)
{
    double *firstVal = (double *)(env->GetDirectBufferAddress(byteBuffer));
    double *lastVal = firstVal + nValues;
    services::SharedPtr<ModelBuilder<double>> *ptr = new services::SharedPtr<ModelBuilder<double>>();
    *ptr = staticPointerCast<ModelBuilder<double>>(*(SharedPtr<ModelBuilder<double>> *)algAddr);
    (*ptr)->setSupportVectors(firstVal, lastVal);
    DAAL_CHECK_THROW((*ptr)->getStatus());
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cSetClassificationCoefficientsFloat
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cSetClassificationCoefficientsFloat
(JNIEnv *env, jobject, jlong algAddr, jobject byteBuffer, jlong nValues)
{
    float *firstVal = (float *)(env->GetDirectBufferAddress(byteBuffer));
    float *lastVal = firstVal + nValues;
    services::SharedPtr<ModelBuilder<float>> *ptr = new services::SharedPtr<ModelBuilder<float>>();
    *ptr = staticPointerCast<ModelBuilder<float>>(*(SharedPtr<ModelBuilder<float>> *)algAddr);
    (*ptr)->setClassificationCoefficients(firstVal, lastVal);
    DAAL_CHECK_THROW((*ptr)->getStatus());
}

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cSetClassificationCoefficientsDouble
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cSetClassificationCoefficientsDouble
(JNIEnv *env, jobject, jlong algAddr, jobject byteBuffer, jlong nValues)
{
    double *firstVal = (double *)(env->GetDirectBufferAddress(byteBuffer));
    double *lastVal = firstVal + nValues;
    services::SharedPtr<ModelBuilder<double>> *ptr = new services::SharedPtr<ModelBuilder<double>>();
    *ptr = staticPointerCast<ModelBuilder<double>>(*(SharedPtr<ModelBuilder<double>> *)algAddr);
    (*ptr)->setClassificationCoefficients(firstVal, lastVal);
    DAAL_CHECK_THROW((*ptr)->getStatus());
}

/*
 * Class:     com_intel_daal_algorithms_svm_ModelBuilder
 * Method:    cSetSupportIndices
 * Signature:(JII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_ModelBuilder_cSetSupportIndices
(JNIEnv *env, jobject, jlong algAddr, jint prec, jobject byteBuffer, jlong nValues)
{
    int *firstVal = (int *)(env->GetDirectBufferAddress(byteBuffer));
    int *lastVal = firstVal + nValues;
    if(prec == 0)
    {
        services::SharedPtr<ModelBuilder<double>> *ptr = new services::SharedPtr<ModelBuilder<double>>();
        *ptr = staticPointerCast<ModelBuilder<double>>(*(SharedPtr<ModelBuilder<double>> *)algAddr);
        (*ptr)->setSupportIndices(firstVal, lastVal);
        DAAL_CHECK_THROW((*ptr)->getStatus());
    }
    else
    {
        services::SharedPtr<ModelBuilder<float>> *ptr = new services::SharedPtr<ModelBuilder<float>>();
        *ptr = staticPointerCast<ModelBuilder<float>>(*(SharedPtr<ModelBuilder<float>> *)algAddr);
        (*ptr)->setSupportIndices(firstVal, lastVal);
        DAAL_CHECK_THROW((*ptr)->getStatus());
    }
}
