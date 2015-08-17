/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
* ************************************************************************/
#pragma once
#ifndef CLSPARSE_BENCHMARK_SpM_SpM_HXX
#define CLSPARSE_BENCHMARK_SpM_SpM_HXX

#include "clSPARSE.h"
#include "clfunc_common.hpp"

// Dummy Code to be deleted later
clsparseStatus clsparseDcsrSpGemm(const clsparseCsrMatrix* sparseMatA, const clsparseCsrMatrix* sparseMatB, clsparseCsrMatrix* sparseMatC, const clsparseControl control)
{
    std::cout << "sparseMat A dimensions = \n";
    std::cout << " rows = " << sparseMatA->num_rows << std::endl;
    std::cout << " cols = " << sparseMatA->num_cols << std::endl;
    std::cout << "nnz = " << sparseMatA->num_nonzeros << std::endl;

    std::cout << "sparseMat B dimensions = \n";
    std::cout << " rows = " << sparseMatB->num_rows << std::endl;
    std::cout << " cols = " << sparseMatB->num_cols << std::endl;
    std::cout << "nnz = " << sparseMatB->num_nonzeros << std::endl;

    return clsparseSuccess;
}//
// Dummy code ends

template <typename T>
class xSpMSpM : public clsparseFunc {
public:
    xSpMSpM(PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type devType) : clsparseFunc(devType, CL_QUEUE_PROFILING_ENABLE), gpuTimer(nullptr), cpuTimer(nullptr)
    {
        //	Create and initialize our timer class, if the external timer shared library loaded
        if (sparseGetTimer)
        {
            gpuTimer = sparseGetTimer(CLSPARSE_GPU);
            gpuTimer->Reserve(1, profileCount);
            gpuTimer->setNormalize(true);

            cpuTimer = sparseGetTimer(CLSPARSE_CPU);
            cpuTimer->Reserve(1, profileCount);
            cpuTimer->setNormalize(true);

            gpuTimerID = gpuTimer->getUniqueID("GPU xSpMSpM", 0);
            cpuTimerID = cpuTimer->getUniqueID("CPU xSpMSpM", 0);
        }
        clsparseEnableAsync(control, false);
    }

    ~xSpMSpM() {}

    void call_func()
    {
        if (gpuTimer && cpuTimer)
        {
            gpuTimer->Start(gpuTimerID);
            cpuTimer->Start(cpuTimerID);

            xSpMSpM_Function(false);

            cpuTimer->Stop(cpuTimerID);
            gpuTimer->Stop(gpuTimerID);
        }
        else
        {
            xSpMSpM_Function(false);
        }
    }//

    double gflops()
    {
        return 0.0;
    }

    std::string gflops_formula()
    {
        return "N/A";
    }

    double bandwidth() // Need to modify this later **********
    {
        //  Assuming that accesses to the vector always hit in the cache after the first access
        //  There are NNZ integers in the cols[ ] array
        //  You access each integer value in row_delimiters[ ] once.
        //  There are NNZ float_types in the vals[ ] array
        //  You read num_cols floats from the vector, afterwards they cache perfectly.
        //  Finally, you write num_rows floats out to DRAM at the end of the kernel.
        return (sizeof(cl_int)*(csrMtx.num_nonzeros + csrMtx.num_rows) + sizeof(T) * (csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows)) / time_in_ns();
    } // end of function

    std::string bandwidth_formula()
    {
        return "GiB/s";
    }// end of function

    void setup_buffer(double pAlpha, double pBeta, const std::string& path)
    {
        sparseFile = path;

        alpha = static_cast<T>(pAlpha);
        beta = static_cast<T>(pBeta);

        // Read sparse data from file and construct a COO matrix from it
        int nnz, row, col;
        clsparseStatus fileError = clsparseHeaderfromFile(&nnz, &row, &col, sparseFile.c_str());
        if (fileError != clsparseSuccess)
            throw clsparse::io_exception("Could not read matrix market header from disk");

        // Now initialise a CSR matrix from the COO matrix
        clsparseInitCsrMatrix(&csrMtx);
        csrMtx.num_nonzeros = nnz;
        csrMtx.num_rows = row;
        csrMtx.num_cols = col;
        clsparseCsrMetaSize(&csrMtx, control);

        cl_int status;
        csrMtx.values = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof(T), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.values");

        csrMtx.colIndices = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof(cl_int), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.colIndices");

        csrMtx.rowOffsets = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY,
            (csrMtx.num_rows + 1) * sizeof(cl_int), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.rowOffsets");

        csrMtx.rowBlocks = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY,
            csrMtx.rowBlockSize * sizeof(cl_ulong), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.rowBlocks");

        if (typeid(T) == typeid(float))
            fileError = clsparseSCsrMatrixfromFile(&csrMtx, sparseFile.c_str(), control);
        else if (typeid(T) == typeid(double))
            fileError = clsparseDCsrMatrixfromFile(&csrMtx, sparseFile.c_str(), control);
        else
            fileError = clsparseInvalidType;

        if (fileError != clsparseSuccess)
            throw clsparse::io_exception("Could not read matrix market data from disk");

        // Create the output CSR Matrix
        clsparseInitCsrMatrix(&csrMtxC);
#if 0
        csrMtxC.num_nonzeros = row * col; // Assuming square matrices (may be = 2*nnz)
        csrMtxC.num_rows = row;
        csrMtxC.num_cols = col;
        clsparseCsrMetaSize(&csrMtxC, control);

        csrMtxC.values = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            csrMtxC.num_nonzeros * sizeof(T), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtxC.values");

        csrMtxC.colIndices = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            csrMtxC.num_nonzeros * sizeof(cl_int), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtxC.colIndices");

        csrMtxC.rowOffsets = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            (csrMtxC.num_rows + 1) * sizeof(cl_int), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtxC.rowOffsets");

        csrMtxC.rowBlocks = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            csrMtxC.rowBlockSize * sizeof(cl_ulong), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtxC.rowBlocks");
#endif

        // Initialize the scalar alpha & beta parameters
        clsparseInitScalar(&a);
        a.value = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                             1 * sizeof(T), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer a.value");

        clsparseInitScalar(&b);
        b.value = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY,
            1 * sizeof(T), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer b.value");

    }// end of function

    void initialize_cpu_buffer()
    {
    }

    void initialize_gpu_buffer()
    {
        CLSPARSE_V(::clEnqueueFillBuffer(queue, a.value, &alpha, sizeof(T), 0,
            sizeof(T) * 1, 0, NULL, NULL), "::clEnqueueFillBuffer alpha.value");

        CLSPARSE_V(::clEnqueueFillBuffer(queue, b.value, &beta, sizeof(T), 0,
            sizeof(T) * 1, 0, NULL, NULL), "::clEnqueueFillBuffer beta.value");

        T scalar_f = 0;

        CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMtxC.values, &scalar_f, sizeof(T), 0,
            sizeof(T) * csrMtxC.num_nonzeros, 0, NULL, NULL), "::clEnqueueFillBuffer csrMtxC.values");

        cl_int scalar_i = 0;
        CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMtxC.colIndices, &scalar_i, sizeof(cl_int), 0,
            sizeof(cl_int) * csrMtxC.num_nonzeros, 0, NULL, NULL), "::clEnqueueFillBuffer csrMtxC.colIndices");

        CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMtxC.rowOffsets, &scalar_i, sizeof(cl_int), 0,
            sizeof(cl_int) * (csrMtxC.num_rows + 1), 0, NULL, NULL), "::clEnqueueFillBuffer csrMtxC.rowOffsets");
    }// end of function

    void reset_gpu_write_buffer()
    {
        T scalar_f  = 0;
        
        CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMtxC.values, &scalar_f, sizeof(T), 0,
            sizeof(T) * csrMtxC.num_nonzeros, 0, NULL, NULL), "::clEnqueueFillBuffer csrMtxC.values");
        
        cl_int scalar_i = 0;
        CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMtxC.colIndices, &scalar_i, sizeof(cl_int), 0,
            sizeof(cl_int) * csrMtxC.num_nonzeros, 0, NULL, NULL), "::clEnqueueFillBuffer csrMtxC.colIndices");

        CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMtxC.rowOffsets, &scalar_i, sizeof(cl_int), 0,
            sizeof(cl_int) * (csrMtxC.num_rows + 1), 0, NULL, NULL), "::clEnqueueFillBuffer csrMtxC.rowOffsets");
    }// end of function

    void read_gpu_buffer()
    {
    }

    void cleanup()
    {
        if (gpuTimer && cpuTimer)
        {
            std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
            size_t sparseBytes = sizeof(cl_int)*(csrMtx.num_nonzeros + csrMtx.num_rows) + sizeof(T) * (csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows);
            cpuTimer->pruneOutliers(3.0);
            cpuTimer->Print(sparseBytes, "GiB/s");
            cpuTimer->Reset();

            gpuTimer->pruneOutliers(3.0);
            gpuTimer->Print(sparseBytes, "GiB/s");
            gpuTimer->Reset();
        }

        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        CLSPARSE_V(::clReleaseMemObject(csrMtx.values),     "clReleaseMemObject csrMtx.values");
        CLSPARSE_V(::clReleaseMemObject(csrMtx.colIndices), "clReleaseMemObject csrMtx.colIndices");
        CLSPARSE_V(::clReleaseMemObject(csrMtx.rowOffsets), "clReleaseMemObject csrMtx.rowOffsets");
        CLSPARSE_V(::clReleaseMemObject(csrMtx.rowBlocks),  "clReleaseMemObject csrMtx.rowBlocks");

        CLSPARSE_V(::clReleaseMemObject(csrMtxC.values),     "clReleaseMemObject csrMtxC.values");
        CLSPARSE_V(::clReleaseMemObject(csrMtxC.colIndices), "clReleaseMemObject csrMtxC.colIndices");
        CLSPARSE_V(::clReleaseMemObject(csrMtxC.rowOffsets), "clReleaseMemObject csrMtxC.rowOffsets");
        //CLSPARSE_V(::clReleaseMemObject(csrMtxC.rowBlocks),  "clReleaseMemObject csrMtxC.rowBlocks");

        CLSPARSE_V(::clReleaseMemObject(a.value), "clReleaseMemObject alpha.value");
        CLSPARSE_V(::clReleaseMemObject(b.value), "clReleaseMemObject beta.value");
    }

private:
    void xSpMSpM_Function(bool flush);

    //  Timers we want to keep
    clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID;
    size_t cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCsrMatrix csrMtx;   // input matrix
    clsparseCsrMatrix csrMtxC;  // Output CSR MAtrix
    clsparseScalar a;
    clsparseScalar b;

    // host values
    T alpha;
    T beta;

    //  OpenCL state
    //cl_command_queue_properties cqProp;
}; // class xSpMSpM

template<> void
xSpMSpM<float>::xSpMSpM_Function(bool flush)
{
    clsparseStatus status = clsparseScsrSpGemm(&csrMtx, &csrMtx, &csrMtxC, control);

    if (flush)
        clFinish(queue);
}// end of single precision function


template<> void
xSpMSpM<double>::xSpMSpM_Function(bool flush)
{
    clsparseStatus status = clsparseDcsrSpGemm(&csrMtx, &csrMtx, &csrMtxC, control);

    if (flush)
        clFinish(queue);
}

#endif // CLSPARSE_BENCHMARK_SpM_SpM_HXX