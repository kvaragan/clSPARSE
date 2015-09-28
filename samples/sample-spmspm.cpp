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
#include <iostream>
#include <vector>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <clSPARSE.h>

/**
* Sample Sparse Matrix times Sparse Matrix multiplication (SPGEMM C++)
*  [C =  A*B]
*
* A - [m x k] matrix in CSR format
* B - [k x n] matrix in CSR format 
* C - [m x n] matrix in CSR format (Output)
*
*
* Program presents usage of clSPARSE library in spgemm (C = A*B) operation
* where A and B are sparse matrix in CSR format.
*
* clSPARSE offers an OpenCL based General Sparse Matrix-Matrix Multiplication 
* Program implemented from 
< See papers:
*  1. Weifeng Liu and Brian Vinter, "A Framework for General Sparse
*      Matrix-Matrix Multiplication on GPUs and Heterogeneous
*      Processors," Journal of Parallel and Distributed Computing, 2015.
*  2. Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
*      Matrix-Matrix Multiplication for Irregular Data," Parallel and
*      Distributed Processing Symposium, 2014 IEEE 28th International
*      (IPDPS '14), pp.370-381, 19-23 May 2014.
*  for details. >
*
* After the matrix is read from disk with the function
* clsparseSCsrMatrixfromFile - 
* Note - Only Single Precision is supported
* Program is executing by completing following steps:
* 1. Setup OpenCL environment
* 2. Setup GPU buffers
* 3. Init clSPARSE library
* 4. Execute algorithm SpMxSpM - 5 API's need to be called
* 5. Shutdown clSPARSE library & OpenCL
*
* usage:
*
* sample-spmspm path/to/first/matrix/in/mtx/format.mtx path/to/second/matrix/in/mtx/format.mtx
*
*/


int main(int argc, char*argv[])
{
    std::string matrixA;
    std::string matrixB;

    if (argc < 3)
    {
        std::cout << "Not Enough Parameters."
            << "Please specify path mtx A and path to mtx  B (two operands of SpGEMM )"
            << std::endl;
    }
    else
    {
        matrixA = std::string(argv[1]);
        matrixB = std::string(argv[2]);
    }

    std::cout << " Executing sample clSPARSE SpM x SpM (C = A * B) C++" << std::endl;

    std::cout << "Matrices will be read from: " << matrixA << " and " << matrixB << std::endl;

    /**  Step 1. Setup OpenCL environment; **/

    // Init OpenCL environment;
    cl_int cl_status;

    // Get OpenCL platforms
    std::vector<cl::Platform> platforms;

    cl_status = cl::Platform::get(&platforms);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting OpenCL platforms"
            << " [" << cl_status << "]" << std::endl;
        return -2;
    }

    int platform_id = 0;
    for (const auto& p : platforms)
    {
        std::cout << "Platform ID " << platform_id++ << " : "
            << p.getInfo<CL_PLATFORM_NAME>() << std::endl;

    }

    // Using first platform
    platform_id = 0;
    cl::Platform platform = platforms[platform_id];

    // Get device from platform
    std::vector<cl::Device> devices;
    cl_status = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting devices from platform"
            << " [" << platform_id << "] " << platform.getInfo<CL_PLATFORM_NAME>()
            << " error: [" << cl_status << "]" << std::endl;
    }

    std::cout << std::endl
              << "Getting devices from platform " << platform_id << std::endl;

    cl_int device_id = 0;
    for (const auto& device : devices)
    {
        std::cout << "Device ID " << device_id++ << " : "
            << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            
    }

    // Using first device;
    device_id = 0;
    cl::Device device = devices[device_id];

    // Create OpenCL Context
    cl::Context context(device);

    // Create OpenCl command queue
    cl::CommandQueue queue(context, device);

    /** Step 2. Setup GPU buffers **/

    clsparseCsrMatrix A;
    clsparseInitCsrMatrix(&A);

    clsparseCsrMatrix B;
    clsparseInitCsrMatrix(&B);

    clsparseCsrMatrix C;
    clsparseInitCsrMatrix(&C);

    /** Step 3. Init clSPARSE library **/
    clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with executing clsparseSetup()" << std::endl;
        return -3;
    }

    // Create clsparseControl object
    clsparseControl control = clsparseCreateControl(queue(), &status);
    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with creating clSPARSE control object"
            << " error [" << status << "]" << std::endl;
        return -4;
    }

    // Read matrices from file
    int nnzA, rowA, colA;

    // read MM header to get the size of the matrix;
    clsparseStatus clstatus =
        clsparseHeaderfromFile(&nnzA, &rowA, &colA, matrixA.c_str());

    if (clstatus != clsparseSuccess)
    {
        std::cout << "Could not read matrix market header A from disk" << std::endl;
        return -5;
    }
    A.num_rows = rowA;
    A.num_cols = colA;
    A.num_nonzeros = nnzA;

    int nnzB, rowB, colB;
    clstatus = clsparseHeaderfromFile(&nnzB, &rowB, &colB, matrixB.c_str());
    if (clstatus != clsparseSuccess)
    {
        std::cout << "Could not read matrix market header A from disk" << std::endl;
        return -5;
    }

    B.num_rows = rowB;
    B.num_cols = colB;
    B.num_nonzeros = nnzB;

    if (A.num_cols != B.num_rows)
    {
        std::cout << "Matrix A & B cannot be multiplied, columns(A) != rows(B)\n";
        return -6;
    }

    // Allocate memory for the input matrices
    A.values = ::clCreateBuffer(context(), CL_MEM_READ_ONLY, 
                              A.num_nonzeros * sizeof(float), NULL, &cl_status);

    A.colIndices = ::clCreateBuffer(context(), CL_MEM_READ_ONLY,
        A.num_nonzeros* sizeof(cl_int), NULL, &cl_status);

    A.rowOffsets = ::clCreateBuffer(context(), CL_MEM_READ_ONLY,
        (A.num_rows + 1)*sizeof(cl_int), NULL, &cl_status);

    B.values = ::clCreateBuffer(context(), CL_MEM_READ_ONLY,
        B.num_nonzeros * sizeof(float), NULL, &cl_status);

    B.colIndices = ::clCreateBuffer(context(), CL_MEM_READ_ONLY,
        B.num_nonzeros* sizeof(cl_int), NULL, &cl_status);

    B.rowOffsets = ::clCreateBuffer(context(), CL_MEM_READ_ONLY,
        (B.num_rows + 1)*sizeof(cl_int), NULL, &cl_status);

    // read the data from the files
    clstatus = clsparseSCsrMatrixfromFile(&A, matrixA.c_str(), control);
    if (clstatus != clsparseSuccess)
    {
        std::cout << "Problem with reading matrix from " << matrixA
            << " Error: " << status << std::endl;
        return -7;
    }

    clstatus = clsparseSCsrMatrixfromFile(&B, matrixB.c_str(), control);
    if (clstatus != clsparseSuccess)
    {
        std::cout << "Problem with reading matrix from " << matrixB
            << " Error: " << status << std::endl;
        return -8;
    }

    // Step 1: Initialize SpMxSpM routine
    clSparseSpGEMM spgemmInfo;
    size_t rowPtrCtSizeInBytes  = 0;
    size_t buffer_d_SizeInBytes = 0;

    clstatus = clsparseSpMSpM_CreateInit(&A, &B, control,
                                        &spgemmInfo,            // output
                                        &rowPtrCtSizeInBytes,   // output
                                        &buffer_d_SizeInBytes); // output


    // Step 2: Computes size of temporary buffers used in CSR SpMxSpM
    int m = A.num_rows;
    cl_mem csrRowPtrCt_d = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, (m + 1) * sizeof(cl_int), NULL, &cl_status);

    /*int pattern = 0;
    clEnqueueFillBuffer(queue(), csrRowPtrCt_d, &pattern, sizeof(cl_int), 0, (m + 1)*sizeof(cl_int), 0, NULL, NULL);*/

    cl_mem buffer_d = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, buffer_d_SizeInBytes, NULL, &cl_status);

    size_t nnzCt = 0;

    clstatus = clsparseSpMSpM_ComputennzCtExt(csrRowPtrCt_d,
                                              rowPtrCtSizeInBytes,
                                              sizeof(cl_int),
                                              buffer_d,
                                              buffer_d_SizeInBytes,
                                              &nnzCt,
                                              spgemmInfo,
                                              control);

    C.rowOffsets = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, (m + 1)*sizeof(cl_int), NULL, &cl_status);

    cl_mem csrColIndCt_d = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, nnzCt*sizeof(cl_int), NULL, &cl_status);
    cl_mem csrValCt_d    = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, nnzCt*sizeof(float),  NULL, &cl_status);
    int nnzC = 0;

    // Step 3: Computes column indices, values and nnz of output CSR matrix C of CSR SpMxSpM
    clstatus = clsparseSpMSpM_ScsrSpmSpmnnzC(C.rowOffsets,          // ( m+1) * sizeof(int) - memory allocated
                                             csrRowPtrCt_d,         // ( m+1) * sizeof(int) - memory allocated
                                             &csrColIndCt_d,        // nnzCt * sizeof(int)
                                             &csrValCt_d,           // nnzCt * sizeof(float) - Single Precision
                                             &nnzC,                    // nnz of Output matrix
                                             spgemmInfo,
                                             control);

    C.colIndices = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, nnzC * sizeof(cl_int), NULL,   &cl_status);
    C.values     = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, nnzC * sizeof(cl_float), NULL, &cl_status);

    // Step 4: Fill sparsity pattern (ColIndices) and values of output CSR C matrix  of SpMxSpM
    clstatus = clsparseSpMSpM_FillScsrOutput(C.rowOffsets,
                                             C.colIndices,
                                             C.values,
                                             csrRowPtrCt_d,
                                             csrColIndCt_d,
                                             csrValCt_d,
                                             nnzC, 
                                             spgemmInfo,
                                             control);


    // Step: 5 Free the resources
    clstatus = clsparseSpMSpM_ReleaseSpmSpm(&spgemmInfo);
    //release mem;
    clReleaseMemObject(A.values);
    clReleaseMemObject(A.colIndices);
    clReleaseMemObject(A.rowOffsets);

    clReleaseMemObject(B.values);
    clReleaseMemObject(B.colIndices);
    clReleaseMemObject(B.rowOffsets);

    clReleaseMemObject(csrRowPtrCt_d);
    clReleaseMemObject(buffer_d);
    clReleaseMemObject(csrColIndCt_d);
    clReleaseMemObject(csrValCt_d);

    clReleaseMemObject(C.values);
    clReleaseMemObject(C.colIndices);
    clReleaseMemObject(C.rowOffsets);
    return 0;
} // End
