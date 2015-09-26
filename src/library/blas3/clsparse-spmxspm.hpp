/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ************************************************************************ */
#pragma once
#ifndef _CLSPARSE_SPMxSPM_HPP__
#define _CLSPARSE_SPMxSPM_HPP__

#include "clSPARSE.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

struct _sparseSpGemm {

    std::vector<int> csrRowPtrCt_h; // row pointers of temporary C matrix
    std::vector<int> csrRowPtrC_h;
    // statistics
    std::vector<int> counter;
    std::vector<int> counter_one;
    std::vector<int> counter_sum;
    std::vector<int> queue_one;
    int nnzCt;

    cl_mem queue_one_d;
    cl_mem csrColIndCt;
    cl_mem csrValCt;

    cl_mem csrRowPtrA;
    cl_mem csrColIndA;
    cl_mem csrValA   ;
    cl_mem csrRowPtrB;
    cl_mem csrColIndB;
    cl_mem csrValB   ;

    int m; /**< Number of rows of Outputs matrix */
    int n; /**< Number of cols of Output matrix  */
};



#endif //_CLSPARSE_SPMxSPM_HPP__