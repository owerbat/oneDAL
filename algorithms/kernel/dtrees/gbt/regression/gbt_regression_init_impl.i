/* file: gbt_regression_init_impl.i */
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
//  Implementation of initialization for gradient boosted trees regression
//  (defaultDense) method.
//--
*/

#include "gbt_regression_init_kernel.h"
#include "dtrees_train_data_helper.i"
#include "threading.h"
#include "gbt_train_aux.i"

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::training::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace init
{
namespace internal
{

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status RegressionInitStep1LocalKernel<algorithmFPType, method, cpu>::compute(
    const NumericTable *x, const NumericTable *y, const HomogenNumericTable<algorithmFPType> * meanDependentVariable,
    const HomogenNumericTable<size_t> * numberOfRows, const HomogenNumericTable<algorithmFPType> * binBorders,
    const HomogenNumericTable<size_t> * binSizes, const Parameter& par)
{
    services::Status s;

    const size_t localMaxBins = par.maxBins;
    const size_t minBinSize = 1;

    // get local mean of dependent variable and number of rows
    const size_t nRows = y->getNumberOfRows();
    daal::internal::ReadColumns<algorithmFPType, cpu> depVarBlock;
    const algorithmFPType* pBlock = depVarBlock.set(const_cast<NumericTable*>(y), 0, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(depVarBlock);

    const size_t nThreads  = daal::threader_get_threads_number();
    const size_t nBlocks   = getNBlocksForOpt<cpu>(nThreads, nRows);
    const bool inParallel  = nBlocks > 1;
    const size_t nPerBlock = nRows / nBlocks;
    const size_t nSurplus  = nRows % nBlocks;
    double sum = 0;
    daal::services::internal::TArray<double, cpu> pSumsArr(nBlocks);
    double * const pSum = pSumsArr.get();
    services::internal::service_memset<double, cpu>(pSum, 0, nBlocks);
    LoopHelper<cpu>::run(inParallel, nBlocks, [&](size_t iBlock) {
        const size_t start = iBlock + 1 > nSurplus ? nPerBlock * iBlock + nSurplus : (nPerBlock + 1) * iBlock;
        const size_t end   = iBlock + 1 > nSurplus ? start + nPerBlock : start + (nPerBlock + 1);
        for (size_t i = start; i < end; ++i)
            pSum[iBlock] += pBlock[i];
    });
    for (size_t i = 0; i < nBlocks; ++i)
        sum += pSum[i];

    algorithmFPType * localMeanDepVarPtr = meanDependentVariable->getArray();
    size_t * localNRowsPtr = numberOfRows->getArray();

    localMeanDepVarPtr[0] = sum / nRows;
    localNRowsPtr[0] = nRows;

    // get binBorders and binSizes
    dtrees::internal::IndexedFeaturesCPU<algorithmFPType, cpu> indexedFeatures;
    dtrees::internal::FeatureTypes featTypes;

    DAAL_CHECK_MALLOC(featTypes.init(*x));

    BinParams prm(localMaxBins, minBinSize);

    DAAL_CHECK_STATUS(s, (indexedFeatures.init(*x, &featTypes, &prm)));

    const size_t nColumns = x->getNumberOfColumns();

    algorithmFPType * localBinBordersPtr = binBorders->getArray();
    size_t * localBinSizesPtr = binSizes->getArray();
    daal::services::internal::EpsilonVal<algorithmFPType> epsilonGetter;
    algorithmFPType epsilon = epsilonGetter.get();
    LoopHelper<cpu>::run(nThreads > 1 ? true : false, nColumns, [&](size_t iCol){
        localBinBordersPtr[iCol] = indexedFeatures.minFeatureValue(iCol)*(1-epsilon) - epsilon;
        for (size_t iBin = 0; iBin < indexedFeatures.numIndices(iCol); ++iBin)
        {
            localBinBordersPtr[(iBin + 1) * nColumns + iCol] = indexedFeatures.binRightBorder(iCol, iBin);
            localBinSizesPtr[iBin * nColumns + iCol] = indexedFeatures.binSize(iCol, iBin);
        }
        for (size_t iBin = indexedFeatures.numIndices(iCol); iBin < localMaxBins; ++iBin)
        {
            localBinBordersPtr[(iBin + 1) * nColumns + iCol] = 0;
            localBinSizesPtr[iBin * nColumns + iCol] = 0;
        }
    });
    return s;
}

template <typename FPType>
FPType getCrossingValue(FPType xMergedBin1, FPType xMergedBin2, FPType xBin1, FPType xBin2, size_t binValue)
{
    FPType max, min;
    if ( xBin2 > xMergedBin2 ) { min = xMergedBin2; } else { min = xBin2; }
    if ( xBin1 > xMergedBin1 ) { max = xBin1; } else { max = xMergedBin1; }
    if ( min - max > 0 )
    {
        FPType value = (FPType)binValue*(min - max)/(xBin2 - xBin1);
        return value;
    }
    else
    {
        return 0;
    }
}

template <typename algorithmFPType, CpuType cpu>
struct BinsMergingTask
{
    DAAL_NEW_DELETE();
    BinsMergingTask(size_t maxBins, size_t nNodes, size_t minBinSize) : _maxBins(maxBins), _nNodes(nNodes), _minBinSize(minBinSize) {}
public:
    bool isValid() const { return true; }
    services::Status merge(HomogenNumericTable<algorithmFPType> **localBinBordersTables,
        HomogenNumericTable<size_t> **localBinSizesTables, size_t iCol, size_t nCols, algorithmFPType * resultBorders, size_t * binsPerFeature)
    {
        services::Status s;
        size_t localMaxBins = localBinSizesTables[0]->getNumberOfRows();
        daal::services::internal::TArray<algorithmFPType, cpu> allBinBordersArray(_nNodes*(localMaxBins+1));
        algorithmFPType * allBinBorders = allBinBordersArray.get();

        // get all bin borders from all nodes
        size_t allBinsCount = 0;
        for (size_t iNode = 0; iNode < _nNodes; ++iNode)
        {
            const algorithmFPType * localBinBordersPtr = localBinBordersTables[iNode]->getArray();
            const size_t * localBinSizesPtr = localBinSizesTables[iNode]->getArray();

            allBinBorders[allBinsCount] = localBinBordersPtr[iCol];
            ++allBinsCount;

            for (size_t iBin = 0; iBin < localMaxBins; ++iBin)
            {
                if (localBinSizesPtr[iBin*nCols + iCol] == 0) break;
                else
                {
                    allBinBorders[allBinsCount] = localBinBordersPtr[(iBin+1)*nCols + iCol];
                    ++allBinsCount;
                }
            }
        }
        // sort borders
        daal::algorithms::internal::qSort<algorithmFPType, cpu>(allBinsCount, allBinBorders);

        // remove duplicates
        daal::services::internal::TArray<algorithmFPType, cpu> mergedBinBordersArray(allBinsCount);
        algorithmFPType * mergedBinBorders = mergedBinBordersArray.get();
        size_t nMergedBins = 0;
        daal::services::internal::MaxVal<algorithmFPType> maxGetter;
        algorithmFPType maxInFPType = maxGetter.get();
        algorithmFPType lastUnique = maxInFPType;
        for (size_t iBin = 0; iBin < allBinsCount; ++iBin)
        {
            if ( allBinBorders[iBin] != lastUnique )
            {
                mergedBinBorders[nMergedBins] = allBinBorders[iBin];
                ++nMergedBins;
                lastUnique = allBinBorders[iBin];
            }
        }
        --nMergedBins; // nBins = nBorders - 1
        if ( nMergedBins > _maxBins )
        {
            // weights assignment
            daal::services::internal::TArray<algorithmFPType, cpu> binWeightsArray(nMergedBins);
            algorithmFPType * binWeights = binWeightsArray.get();
            for (size_t i = 0; i < nMergedBins; ++i) { binWeights[i] = 0.; }

            for (size_t iMergedBin = 0; iMergedBin < nMergedBins; ++iMergedBin)
            {
                for (size_t iNode = 0; iNode < _nNodes; ++iNode)
                {
                    const algorithmFPType * binsBorders = localBinBordersTables[iNode]->getArray() + iCol;
                    const size_t * binSizes = localBinSizesTables[iNode]->getArray() + iCol;

                    for (size_t iBin = 0; iBin < localMaxBins; ++iBin)
                    {
                        if ( binSizes[iBin * nCols] == 0 ) break;
                        if ( binsBorders[iBin * nCols] >= mergedBinBorders[iMergedBin+1] ) break;

                        algorithmFPType weight = getCrossingValue<algorithmFPType>(
                            mergedBinBorders[iMergedBin], mergedBinBorders[iMergedBin+1],
                            binsBorders[iBin * nCols], binsBorders[(iBin+1) * nCols], binSizes[iBin * nCols]);
                        binWeights[iMergedBin] += weight;
                    }
                }
            }
            // bins merging
            while ( nMergedBins > _maxBins )
            {
                size_t minIdx;
                algorithmFPType minSumWeights = maxInFPType;
                for (size_t i = 0; i < nMergedBins - 1; ++i)
                {
                    if ( binWeights[i] + binWeights[i+1] < minSumWeights )
                    {
                        minIdx = i;
                        minSumWeights = binWeights[i] + binWeights[i+1];
                    }
                }
                binWeights[minIdx] += binWeights[minIdx+1];
                for (size_t i = minIdx + 1; i < nMergedBins; ++i)
                {
                    mergedBinBorders[i] = mergedBinBorders[i+1];
                }
                for (size_t i = minIdx + 1; i < nMergedBins-1; ++i)
                {
                    binWeights[i] = binWeights[i+1];
                }
                nMergedBins--;
            }

            // continue merge bins if some < minBinSize
            size_t iBin = 0;
            algorithmFPType minBinSize = _minBinSize;
            while ( iBin < nMergedBins )
            {
                if ( binWeights[iBin] < minBinSize )
                {
                    bool addToRight;
                    if ( iBin == nMergedBins - 1 ) addToRight = false;
                    else if ( iBin == 0 ) addToRight = true;
                    else if ( binWeights[iBin - 1] < binWeights[iBin + 1] ) addToRight = false;
                    else addToRight = true;
                    if ( addToRight )
                    {
                        binWeights[iBin + 1] += binWeights[iBin];
                        for (size_t jBin = iBin; jBin < nMergedBins - 1; ++jBin)
                        {
                            binWeights[jBin] = binWeights[jBin + 1];
                        }
                        for (size_t jBin = iBin; jBin < nMergedBins; ++jBin)
                        {
                            mergedBinBorders[jBin] = mergedBinBorders[jBin + 1];
                        }
                    }
                    else
                    {
                        binWeights[iBin] += binWeights[iBin - 1];
                        for (size_t jBin = iBin - 1; jBin < nMergedBins - 1; ++jBin)
                        {
                            binWeights[jBin] = binWeights[jBin + 1];
                        }
                        for (size_t jBin = iBin - 1; jBin < nMergedBins; ++jBin)
                        {
                            mergedBinBorders[jBin] = mergedBinBorders[jBin + 1];
                        }
                    }
                    nMergedBins--;
                }
                else
                {
                    ++iBin;
                }
            }
        }
        binsPerFeature[0] = nMergedBins;
        for (size_t iBin = 0; iBin < nMergedBins; ++iBin)
        {
            resultBorders[iBin * nCols] = mergedBinBorders[iBin + 1];
        }
        return s;
    }
private:
    const size_t _nNodes;
    const size_t _maxBins;
    const size_t _minBinSize;
};

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status RegressionInitStep2MasterKernel<algorithmFPType, method, cpu>::compute(
    size_t nNodes, const DataCollectionPtr localNumberOfRows, const DataCollectionPtr localMeanDepVars,
    const DataCollectionPtr localBinBorders, const DataCollectionPtr localBinSizes,
    HomogenNumericTable<algorithmFPType> * ntInitialResponse, const HomogenNumericTable<algorithmFPType> * mergedBinBorders,
    const HomogenNumericTable<size_t> * binQuantities,
    DataCollection *dcBinValues, const Parameter& par)
{
    const size_t maxBins = par.maxBins;
    const size_t minBinSize = par.minBinSize;

    daal::services::internal::TArray<HomogenNumericTable<algorithmFPType>*, cpu> localBinBordersTablesArr(nNodes);
    daal::services::internal::TArray<HomogenNumericTable<size_t>*, cpu> localBinSizesTablesArr(nNodes);
    HomogenNumericTable<algorithmFPType> ** localBinBordersTables = localBinBordersTablesArr.get();
    HomogenNumericTable<size_t> ** localBinSizesTables = localBinSizesTablesArr.get();
    for (size_t iNode = 0; iNode < nNodes; ++iNode)
    {
        localBinBordersTables[iNode] = HomogenNumericTable<algorithmFPType>::cast((*localBinBorders)[iNode]).get();
        localBinSizesTables[iNode] = static_cast<HomogenNumericTable<size_t>*>(((*localBinSizes)[iNode]).get());
    }

    size_t nCols = localBinBordersTables[0]->getNumberOfColumns();

    // find global mean
    double mean = 0;
    size_t nRows = 0;
    for (size_t iNode = 0; iNode < nNodes; ++iNode)
    {
        NumericTable *localMeanNTPtr = (NumericTable::cast((*localMeanDepVars)[iNode])).get();
        daal::internal::ReadRows<algorithmFPType, cpu> localMeanRows(localMeanNTPtr, 0, 1);
        algorithmFPType localMean = localMeanRows.get()[0];

        NumericTable *localNRowsNTPtr = (NumericTable::cast((*localNumberOfRows)[iNode])).get();
        daal::internal::ReadRows<int, cpu> localNRowsRows(localNRowsNTPtr, 0, 1);
        size_t localNRows = static_cast<size_t>(localNRowsRows.get()[0]);

        mean += localMean*localNRows;
        nRows += localNRows;
    }
    mean /= nRows;

    ntInitialResponse->assign(mean);

    // merging
    typedef BinsMergingTask<algorithmFPType, cpu> MergingTask;
    typedef MergingTask TlsTask;

    daal::tls<TlsTask*> tlsData([=]()->TlsTask*
    {
        TlsTask* res = new MergingTask(maxBins, nNodes, minBinSize);
        if(res && !res->isValid())
        {
            delete res;
            res = nullptr;
        }
        return res;
    });

    daal::services::internal::TArray<algorithmFPType*, cpu> bordersArray(nCols);
    daal::services::internal::TArray<size_t*, cpu> binQuantitiesArray(nCols);
    algorithmFPType ** bordersPtrs = bordersArray.get();
    size_t ** binQuantitiesPtrs = binQuantitiesArray.get();
    for (size_t iCol = 0; iCol < nCols; ++iCol)
    {
        bordersPtrs[iCol] = mergedBinBorders->getArray() + iCol;
        binQuantitiesPtrs[iCol] = binQuantities->getArray() + iCol;
    }

    SafeStatus safeStat;
    daal::threader_for(nCols, nCols, [&](size_t iCol)
    {
        TlsTask* task = tlsData.local();
        DAAL_CHECK_THR(task, services::ErrorMemoryAllocationFailed);
        safeStat |= task->merge(localBinBordersTables, localBinSizesTables, iCol, nCols, bordersPtrs[iCol], binQuantitiesPtrs[iCol]);
    });
    tlsData.reduce([&](TlsTask* task)-> void
    {
        delete task;
    });

    dcBinValues->clear();

    for (size_t i = 0; i < nCols; ++i)
    {
        services::Status s;
        services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntBinValues = HomogenNumericTable<algorithmFPType>::create(binQuantitiesPtrs[i][0], 1, NumericTable::doAllocate, &s);
        algorithmFPType * const binValues = ntBinValues->getArray();
        for (size_t j = 0; j < binQuantitiesPtrs[i][0]; ++j)
        {
            binValues[j] = bordersPtrs[i][j * nCols];
        }
        dcBinValues->push_back(ntBinValues);
    }
    return safeStat.detach();
}

template <typename algorithmFPType, typename BinnedDataType, Method method, CpuType cpu>
services::Status step3ComputeImpl(
    const HomogenNumericTable<algorithmFPType> *mergedBinBorders, const HomogenNumericTable<size_t> *binQuantities,
    const NumericTable *x, const HomogenNumericTable<BinnedDataType> *binnedData,
    const HomogenNumericTable<BinnedDataType> *transposedBinnedData,
    const HomogenNumericTable<algorithmFPType> *ntInitialResponse,
    HomogenNumericTable<algorithmFPType> *ntResponse,
    HomogenNumericTable<int> *ntTreeOrder,
    const Parameter& par)
{
    const size_t nRows = x->getNumberOfRows();
    const size_t nFeatures = x->getNumberOfColumns();
    const size_t maxBins = par.maxBins;

    ntResponse->assign(ntInitialResponse->getArray()[0]);
    for (size_t i = 0; i < nRows; ++i)
    {
        ntTreeOrder->getArray()[i] = i;
    }

    TArray<size_t, cpu> binSizes(nFeatures);
    for (size_t iCol = 0; iCol < nFeatures; ++iCol)
    {
        binSizes[iCol] = (binQuantities->getArray())[iCol];
    }

    daal::internal::ReadRows<algorithmFPType, cpu> dataRows(const_cast<NumericTable*>(x), 0, nRows);
    const algorithmFPType * const data = dataRows.get();

    BinnedDataType * const binnedDataPtr = binnedData->getArray();
    BinnedDataType * const transposedBinnedDataPtr = transposedBinnedData->getArray();

    const algorithmFPType * const bordersPtr = mergedBinBorders->getArray();

    TArray<algorithmFPType, cpu> binBordersArray(nFeatures * maxBins);
    for (size_t i = 0; i < nFeatures; ++i)
    {
        for (size_t j = 0; j < binSizes[i]; ++j)
        {
            binBordersArray[i * maxBins + j] = bordersPtr[j * nFeatures + i];
        }
    }

    const size_t blockSize = 2048;
    size_t nBlocks = nRows / blockSize;
    nBlocks += !!(nRows - nBlocks * blockSize);

    LoopHelper<cpu>::run(true, nBlocks, [&](size_t block)
    {
        const size_t iStart = block * blockSize;
        const size_t iEnd = (((block + 1) * blockSize > nRows) ? nRows : iStart + blockSize);

        for (size_t iRow = iStart; iRow < iEnd; ++iRow)
        {
            for (size_t iCol = 0; iCol < nFeatures; ++iCol)
            {
                const algorithmFPType * const curBorders = binBordersArray.get() + iCol * maxBins;
                const size_t curNBins = binSizes[iCol];
                const algorithmFPType value = data[iRow * nFeatures + iCol];
                size_t right = curNBins - 1;
                size_t left = 0;
                while (left < right)
                {
                    size_t center = (right + left) / 2;

                    if (value > curBorders[center])
                    {
                        left = center + 1;
                    }
                    else
                    {
                        right = center;
                    }
                }
                binnedDataPtr[iRow * nFeatures + iCol] = (BinnedDataType)left;
                transposedBinnedDataPtr[iCol * nRows + iRow] = (BinnedDataType)left;
            }
        }
    });

    return services::Status();
};

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status RegressionInitStep3LocalKernel<algorithmFPType, method, cpu>::compute(
    const HomogenNumericTable<algorithmFPType> *mergedBinBorders, const HomogenNumericTable<size_t> *binQuantities, const NumericTable *x,
    const HomogenNumericTable<algorithmFPType> *ntInitialResponse, const DistributedPartialResultStep3 *partialResult, const Parameter& par)
{
    HomogenNumericTable<algorithmFPType> *ntResponse = dynamic_cast<HomogenNumericTable<algorithmFPType>*>((partialResult->get(step3Response)).get());
    HomogenNumericTable<int> *ntTreeOrder = dynamic_cast<HomogenNumericTable<int>*>((partialResult->get(step3TreeOrder)).get());

    const size_t maxBins = par.maxBins;
    if (maxBins <= 256)
    {
        HomogenNumericTable<uint8_t> *binnedData = dynamic_cast<HomogenNumericTable<uint8_t>*>((partialResult->get(gbt::regression::init::step3BinnedData)).get());
        HomogenNumericTable<uint8_t> *transposedBinnedData = dynamic_cast<HomogenNumericTable<uint8_t>*>((partialResult->get(gbt::regression::init::step3TransposedBinnedData)).get());
        return step3ComputeImpl<algorithmFPType, uint8_t, method, cpu>(mergedBinBorders, binQuantities, x, binnedData, transposedBinnedData, ntInitialResponse, ntResponse, ntTreeOrder, par);
    }
    else if (maxBins <= 65536)
    {
        HomogenNumericTable<uint16_t> *binnedData = dynamic_cast<HomogenNumericTable<uint16_t>*>((partialResult->get(gbt::regression::init::step3BinnedData)).get());
        HomogenNumericTable<uint16_t> *transposedBinnedData = dynamic_cast<HomogenNumericTable<uint16_t>*>((partialResult->get(gbt::regression::init::step3TransposedBinnedData)).get());
        return step3ComputeImpl<algorithmFPType, uint16_t, method, cpu>(mergedBinBorders, binQuantities, x, binnedData, transposedBinnedData, ntInitialResponse, ntResponse, ntTreeOrder, par);
    }
    else
    {
        HomogenNumericTable<int> *binnedData = dynamic_cast<HomogenNumericTable<int>*>((partialResult->get(gbt::regression::init::step3BinnedData)).get());
        HomogenNumericTable<int> *transposedBinnedData = dynamic_cast<HomogenNumericTable<int>*>((partialResult->get(gbt::regression::init::step3TransposedBinnedData)).get());
        return step3ComputeImpl<algorithmFPType, int, method, cpu>(mergedBinBorders, binQuantities, x, binnedData, transposedBinnedData, ntInitialResponse, ntResponse, ntTreeOrder, par);
    }
}

} // namespace internal
}
}
}
}
} // namespace daal
