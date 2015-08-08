#ifndef _SPARSE_MATRIX_ENVIRONMENT_H_
#define _SPARSE_MATRIX_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <clSPARSE.h>

#include "clsparse_environment.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
using CLSE = ClSparseEnvironment;

namespace uBLAS = boost::numeric::ublas;

/**
* @brief The CSRSparseEnvironment class will have the input parameters for SpMSpM tests
* They are list of csr matrices in csr format in mtx files.
*/
// Currently only single precision is considered
class CSRSparseEnvironment : public ::testing::Environment {
public:
    using sMatrixType = uBLAS::compressed_matrix<float,  uBLAS::row_major, 0, uBLAS::unbounded_array<int> >;
    using dMatrixType = uBLAS::compressed_matrix<double, uBLAS::row_major, 0, uBLAS::unbounded_array<size_t> >;

    explicit CSRSparseEnvironment(const std::string& path, cl_command_queue queue, cl_context context)
        : queue(queue), context(context)
    {
        file_name = path;
        clsparseStatus read_status = clsparseHeaderfromFile(&n_vals, &n_rows, &n_cols, file_name.c_str());
        if (read_status)
        {
            exit(-3);
        }

        clsparseInitCsrMatrix(&csrSMatrix);
        csrSMatrix.num_nonzeros = n_vals;
        csrSMatrix.num_rows = n_rows;
        csrSMatrix.num_cols = n_cols;
        clsparseCsrMetaSize(&csrSMatrix, CLSE::control);

        //  Load single precision data from file; this API loads straight into GPU memory
        cl_int status;
        csrSMatrix.values = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrix.num_nonzeros * sizeof(cl_float), NULL, &status);

        csrSMatrix.colIndices = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrix.num_nonzeros * sizeof(cl_int), NULL, &status);

        csrSMatrix.rowOffsets = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            (csrSMatrix.num_rows + 1) * sizeof(cl_int), NULL, &status);

        csrSMatrix.rowBlocks = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrix.rowBlockSize * sizeof(cl_ulong), NULL, &status);

        clsparseStatus fileError = clsparseSCsrMatrixfromFile(&csrSMatrix, file_name.c_str(), CLSE::control);
        if (fileError != clsparseSuccess)
            throw std::runtime_error("Could not read matrix market data from disk");

        //reassign the new matrix dimmesnions calculated clsparseCCsrMatrixFromFile to global variables
        /*n_vals = csrSMatrix.num_nonzeros;
        n_cols = csrSMatrix.num_cols;
        n_rows = csrSMatrix.num_rows;*/

        //  Download sparse matrix data to host
        //  First, create space on host to hold the data
        ublasSCsr = sMatrixType(n_rows, n_cols, n_vals);

        // This is nasty. Without that call ublasSCsr is not working correctly.
        ublasSCsr.complete_index1_data();

        // copy host matrix arrays to device;
        cl_int copy_status;

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.values, CL_TRUE, 0,
            csrSMatrix.num_nonzeros * sizeof(cl_float),
            ublasSCsr.value_data().begin(),
            0, NULL, NULL);

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.rowOffsets, CL_TRUE, 0,
            (csrSMatrix.num_rows + 1) * sizeof(cl_int),
            ublasSCsr.index1_data().begin(),
            0, NULL, NULL);

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.colIndices, CL_TRUE, 0,
            csrSMatrix.num_nonzeros * sizeof(cl_int),
            ublasSCsr.index2_data().begin(),
            0, NULL, NULL);

        if (copy_status)
        {
            TearDown();
            exit(-5);
        }
    }// end C'tor

    void SetUp()
    {
        // Prepare data to it's default state
    }

    //cleanup
    void TearDown()
    {
    }

    std::string getFileName()
    {
        return file_name;
    }

    ~CSRSparseEnvironment()
    {
        //release buffers;
        ::clReleaseMemObject(csrSMatrix.values);
        ::clReleaseMemObject(csrSMatrix.colIndices);
        ::clReleaseMemObject(csrSMatrix.rowOffsets);
        ::clReleaseMemObject(csrSMatrix.rowBlocks);

        //bring csrSMatrix  to its initial state
        clsparseInitCsrMatrix(&csrSMatrix);
    }
        

    static sMatrixType ublasSCsr;
    //static sMatrixType ublasCsrB;
    //static sMatrixType ublasCsrC;    

    static cl_int n_rows;
    static cl_int n_cols;
    static cl_int n_vals;

    //cl buffers ;
    static clsparseCsrMatrix csrSMatrix; // input 1
    //static clsparseCsrMatrix csrMatrixB; // input 2
    //static clsparseCsrMatrix csrMatrixC; // output

    static std::string file_name;

private:
    cl_command_queue queue;
    cl_context context;
};


#endif // _SPARSE_MATRIX_ENVIRONMENT_H_