/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * Copyright 2015 Vratis, Ltd.
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
#ifndef _CL_SPARSE_H_
#define _CL_SPARSE_H_

// CMake-generated file to define export related preprocessor macros
#include "clsparse_export.h"

#ifdef __cplusplus
extern "C" {
#endif

    // Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION < 200 )
#include "clSPARSE-1x.h"
#else
#include "clSPARSE-2x.h"
#endif

    typedef enum clsparseStatus_ {
        clsparseSuccess = CL_SUCCESS,
        clsparseInvalidValue = CL_INVALID_VALUE,
        clsparseInvalidCommandQueue = CL_INVALID_COMMAND_QUEUE,
        clsparseInvalidContext = CL_INVALID_CONTEXT,
        clsparseInvalidMemObject = CL_INVALID_MEM_OBJECT,
        clsparseInvalidDevice = CL_INVALID_DEVICE,
        clsparseInvalidEventWaitList = CL_INVALID_EVENT_WAIT_LIST,
        clsparseInvalidEvent = CL_INVALID_EVENT,
        clsparseOutOfResources = CL_OUT_OF_RESOURCES,
        clsparseOutOfHostMemory = CL_OUT_OF_HOST_MEMORY,
        clsparseInvalidOperation = CL_INVALID_OPERATION,
        clsparseCompilerNotAvailable = CL_COMPILER_NOT_AVAILABLE,
        clsparseBuildProgramFailure = CL_BUILD_PROGRAM_FAILURE,
        clsparseInvalidKernelArgs = CL_INVALID_KERNEL_ARGS,

        /* Extended error codes */
        clsparseNotImplemented = -1024, /**< Functionality is not implemented */
        clsparseNotInitialized,                 /**< clsparse library is not initialized yet */
        clsparseStructInvalid,                 /**< clsparse library is not initialized yet */
        clsparseInvalidSize,                    /**< Invalid size of object > */
        clsparseInvalidMemObj,                  /**< Checked obejct is no a valid cl_mem object */
        clsparseInsufficientMemory,             /**< The memory object for vector is too small */
        clsparseInvalidControlObject,           /**< clsparseControl object is not valid */
        clsparseInvalidFile,                    /**< Error reading the sparse matrix file */
        clsparseInvalidFileFormat,              /**< Only specific documented sparse matrix files supported */
        clsparseInvalidKernelExecution,          /**< Problem with kenrel execution */
        clsparseInvalidType,                     /** < Wrong type provided > */

        /* Solver control */
        clsparseInvalidSolverControlObject = -2048,
        clsparseInvalidSystemSize,
        clsparseIterationsExceeded,
        clsparseToleranceNotReached,
        clsparseSolverError,

        /* SpMxSpM */
        clsparseInvalidMatrixDimensions = -3048,  /**< Input matrices cannot be multiplied - cols of A != rows of B */
    } clsparseStatus;


    // clsparseControl keeps the data relevant for
    // OpenCL operations like kernel execution, mem alocation, sync.
    /* To be considered:
        - how the allocation should look like?
        IMO clsparseControl ctrl = clsparseControl { .queue = queue ... } is not nice
        - if there is sth like clsparseControl how we should destroy it? in the tearDown function?
        - if the user call the clReleaseCommandQueue the clsparseControl become invalid.
        */
    //
    typedef struct _clsparseControl*  clsparseControl;

    //setup the control from external queue;
    CLSPARSE_EXPORT clsparseControl
        clsparseCreateControl( cl_command_queue queue, clsparseStatus *status );

    //enable/disable asynchronous behavior for clSPARSE;
    CLSPARSE_EXPORT clsparseStatus
        clsparseEnableAsync( clsparseControl control, cl_bool async );

    //enable/disable the use of compensated summation
    CLSPARSE_EXPORT clsparseStatus
        clsparseEnableExtendedPrecision( clsparseControl control, cl_bool async );

    //setup events to sync
    //TODO:: NOT WORKING! NDRange throws Failure
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetupEventWaitList( clsparseControl control,
        cl_uint num_events_in_wait_list,
        cl_event* event_wait_list );

    //get the event from the last kernel execution
    CLSPARSE_EXPORT clsparseStatus
        clsparseGetEvent( clsparseControl control, cl_event* event );

    // just sets the fields to 0 or Null and free allocated struc.
    // We do not own the queue, context, etc;
    CLSPARSE_EXPORT clsparseStatus
        clsparseReleaseControl( clsparseControl control );

    /*
     * Solver control: Object controlling the solver execution
     */
    typedef enum _print_mode{
        QUIET = 0,
        NORMAL,
        VERBOSE
    } PRINT_MODE;

    typedef enum _precond
    {
        NOPRECOND = 0,
        DIAGONAL
    } PRECONDITIONER;

    typedef struct _solverControl*  clSParseSolverControl;

    CLSPARSE_EXPORT clSParseSolverControl
    clsparseCreateSolverControl(PRECONDITIONER precond, cl_int maxIters,
                            cl_double relTol, cl_double absTol);

    CLSPARSE_EXPORT clsparseStatus
        clsparseReleaseSolverControl( clSParseSolverControl solverControl );

    //here maybe some other solver control utils;
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetSolverParams( clSParseSolverControl solverControl,
                        PRECONDITIONER precond,
                        cl_int maxIters, cl_double relTol, cl_double absTol);

    CLSPARSE_EXPORT clsparseStatus
        clsparseSolverPrintMode( clSParseSolverControl solverControl, PRINT_MODE mode );

    /* Conjugate Gradients solver */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrcg( cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
        clSParseSolverControl solverControl, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseDcsrcg(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control);

/* Bi-Conjugate Gradients Stabilized solver */
CLSPARSE_EXPORT clsparseStatus
clsparseScsrbicgStab(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsrbicgStab(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control);
    //  Library initialization and deconstruction functions
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetup( void );

    CLSPARSE_EXPORT clsparseStatus
        clsparseTeardown( void );

    CLSPARSE_EXPORT clsparseStatus
        clsparseGetVersion( cl_uint *major, cl_uint *minor, cl_uint *patch, cl_uint *tweak );

    // Convenience sparse matrix construction functions
    CLSPARSE_EXPORT clsparseStatus
        clsparseInitScalar( clsparseScalar* scalar );

    CLSPARSE_EXPORT clsparseStatus
        clsparseInitVector( cldenseVector* vec );

    CLSPARSE_EXPORT clsparseStatus
        clsparseInitCooMatrix( clsparseCooMatrix* cooMatx );

    CLSPARSE_EXPORT clsparseStatus
        clsparseInitCsrMatrix( clsparseCsrMatrix* csrMatx );

    CLSPARSE_EXPORT clsparseStatus
        cldenseInitMatrix( cldenseMatrix* denseMatx );

    // Convenience functions provided by library to read sparse matrices from file
    CLSPARSE_EXPORT clsparseStatus
        clsparseHeaderfromFile( cl_int* nnz, cl_int* row, cl_int* col, const char* filePath );

    CLSPARSE_EXPORT clsparseStatus
        clsparseSCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseDCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, clsparseControl control );
//    CLSPARSE_EXPORT clsparseStatus
//        clsparseCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

    // Functions needed to compute SpM-dV operations with CSR-adaptive algorithms
    CLSPARSE_EXPORT clsparseStatus
        clsparseCsrMetaSize( clsparseCsrMatrix* csrMatx, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseSCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseDCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseCsrMetaCompute( clsparseCsrMatrix* csrMatx, clsparseControl control );

    /* BLAS 1 routines for dense vector*/

    /* SCALE r = alpha * y */

    CLSPARSE_EXPORT clsparseStatus
        cldenseSscale(cldenseVector* r,
        const clsparseScalar* alpha,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDscale( cldenseVector* r,
        const clsparseScalar* alpha,
        const cldenseVector* y,
        const clsparseControl control );

    /* AXPY: r = alpha*x + y */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSaxpy( cldenseVector* r,
        const clsparseScalar* alpha, const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDaxpy( cldenseVector* r,
        const clsparseScalar* alpha, const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    /* AXPY: r = alpha*x + beta*y */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSaxpby( cldenseVector* r,
        const clsparseScalar* alpha, const cldenseVector* x,
        const clsparseScalar* beta,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDaxpby( cldenseVector* r,
        const clsparseScalar* alpha, const cldenseVector* x,
        const clsparseScalar* beta,
        const cldenseVector* y,
        const clsparseControl control );

    /* Reduce (sum) */
    CLSPARSE_EXPORT clsparseStatus
        cldenseIreduce( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseSreduce( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDreduce( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    /* norms */

    /* L1 norm */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSnrm1( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDnrm1( clsparseScalar *s,
        const cldenseVector* x,
        const clsparseControl control );

    /* L2 norm */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSnrm2( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDnrm2( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    /* dot product */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSdot( clsparseScalar* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDdot( clsparseScalar* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    /* elementwise operations for dense vectors +, -, *, / */

    // +
    CLSPARSE_EXPORT clsparseStatus
        cldenseSadd( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDadd( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );
    // -
    CLSPARSE_EXPORT clsparseStatus
        cldenseSsub( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDsub( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    // *
    CLSPARSE_EXPORT clsparseStatus
        cldenseSmul( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDmul( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );
    // /
    CLSPARSE_EXPORT clsparseStatus
        cldenseSdiv( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDdiv( cldenseVector* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    // BLAS 2 routines
    // SpM-dV
    // y = \alpha * A * x + \beta * y

    //new possible implementation of csrmv with control object
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrmv( const clsparseScalar* alpha,
        const clsparseCsrMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsrmv( const clsparseScalar* alpha,
        const clsparseCsrMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );


    CLSPARSE_EXPORT clsparseStatus
        clsparseScoomv( const clsparseScalar* alpha,
        const clsparseCooMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseDcoomv( const clsparseScalar* alpha,
        const clsparseCooMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );

    // Sparse BLAS 3 routines
    // SpM-dM
    // C = \alpha * A * B  + \beta * C
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrmm( const clsparseScalar* alpha,
        const clsparseCsrMatrix* sparseMatA,
        const cldenseMatrix* denseMatB,
        const clsparseScalar* beta,
        cldenseMatrix* denseMatC,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsrmm( const clsparseScalar* alpha,
        const clsparseCsrMatrix* sparseMatA,
        const cldenseMatrix* denseMatB,
        const clsparseScalar* beta,
        cldenseMatrix* denseMatC,
        const clsparseControl control );

/* Matrix conversion routines */
// Input matrix have to be sorted by row and col.
// The clsparse reading routines guarantee that

//CSR to COO
CLSPARSE_EXPORT clsparseStatus
clsparseScsr2coo(const clsparseCsrMatrix* csr,
                 clsparseCooMatrix* coo,
                 const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsr2coo(const clsparseCsrMatrix* csr,
                 clsparseCooMatrix* coo,
                 const clsparseControl control);

// COO to CSR
CLSPARSE_EXPORT clsparseStatus
clsparseScoo2csr(const clsparseCooMatrix* coo,
                 clsparseCsrMatrix* csr,
                 const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcoo2csr(const clsparseCooMatrix* coo,
                 clsparseCsrMatrix* csr,
                 const clsparseControl control);

//CSR 2 Dense
CLSPARSE_EXPORT clsparseStatus
clsparseScsr2dense(const clsparseCsrMatrix* csr,
                   cldenseMatrix* A,
                   const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsr2dense(const clsparseCsrMatrix* csr,
                   cldenseMatrix* A,
                   clsparseControl control);

//Dense to CSR
CLSPARSE_EXPORT clsparseStatus
clsparseSdense2csr(const cldenseMatrix* A,
                   clsparseCsrMatrix* csr,
                   const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDdense2csr(const cldenseMatrix* A, clsparseCsrMatrix* csr,
                   const clsparseControl control);

typedef struct _sparseSpGemm* clSparseSpGEMM;
  /*!
   * \brief Single Precision CSR Sparse Matrix times Sparse Matrix
   * \details \f$ C \leftarrow A \ast B \f$
   * \param[in] sparseMatA Input CSR sparse matrix
   * \param[in] sparseMatB Input CSR sparse matrix
   * \param[out] sparseMatC Output CSR sparse matrix
   * \param[in] control A valid clsparseControl created with clsparseCreateControl
   *
   * \ingroup BLAS-3
   */
 CLSPARSE_EXPORT clsparseStatus
        clsparseScsrSpGemm(
        const clsparseCsrMatrix* sparseMatA,
        const clsparseCsrMatrix* sparseMatB,
              clsparseCsrMatrix* sparseMatC,
        const clsparseControl control );

 /*!
 * \brief Creates auxilary data structure for  CSR Sparse Matrix times Sparse Matrix operation
 * \details \f$ C \leftarrow A \ast B \f$ - Step 1
 * \param[in] sparseMatA Input CSR sparse matrix
 * \param[in] sparseMatB Input CSR sparse matrix
 * \param[in] control A valid clsparseControl created with clsparseCreateControl
 * \param[out] spgemmInfo clSparseSpGEMM auxiliary data structre is created
 * \param[out] rowPtrCtSizeInBytes - Size in Bytes for the temporary device to be allocated
 * \param[out] buffer_d_SizeInBytes - Size in Bytes for the the device memory to be allocated
 * \ingroup BLAS-3
 */
 CLSPARSE_EXPORT clsparseStatus
     clsparseSpMSpM_CreateInit(const clsparseCsrMatrix* sparseMatA,
                              const clsparseCsrMatrix* sparseMatB,
                              const clsparseControl control, 
                              clSparseSpGEMM* spgemmInfo,  
                              size_t* rowPtrCtSizeInBytes, 
                              size_t* buffer_d_SizeInBytes);
 /*!
 * \brief Free auxilary data structure used in CSR Sparse Matrix times Sparse Matrix operation
 * \details \f$ C \leftarrow A \ast B \f$
 * \param[in] spgemmInfo clSparseSpGEMM auxiliary data structre created with clsparseCreateInitSpmSpm
 * \ingroup BLAS-3
 */
 CLSPARSE_EXPORT clsparseStatus
     clsparseSpMSpM_ReleaseSpmSpm(clSparseSpGEMM* spgemmInfo);

 /*!
 * \brief Computes size of temporary buffers used in CSR Sparse Matrix times Sparse Matrix operation
 * \details \f$ C \leftarrow A \ast B \f$ Step 2
 * \param[in] csrRowPtrCt_d Device memory of size rowPtrCtSizeInBytes
 * \param[in] rowPtrCtSizeInBytes Size in Bytes of csrRowPtrCt_d
 * \param[in] singleElemSize - Size in Bytes of single element of csrRowPtrCt_d
 * \param[in] buffer_d device memory of size buffer_d_SizeInBytes
 * \param[in] buffer_d_SizeInBytes size in bytes of the device memory buffer_d
 * \param[out] nnzCt - stores nnz of Ct 
 * \param[in] spgemmInfo clSparseSpGEMM auxiliary data structre created with clsparseCreateInitSpmSpm
 * \param[in] control A valid clsparseControl created with clsparseCreateControl
 * \ingroup BLAS-3
 */
 CLSPARSE_EXPORT clsparseStatus
     clsparseSpMSpM_ComputennzCtExt(cl_mem csrRowPtrCt_d,
                                    size_t rowPtrCtSizeInBytes,
                                    size_t singleElemSize,
                                    cl_mem buffer_d,
                                    size_t buffer_d_SizeInBytes,
                                    size_t *nnzCt,               
                                    clSparseSpGEMM spgemmInfo,
                                    const clsparseControl control);

 /*!
 * \brief Computes column indices, values and nnz of output CSR matrix C of CSR Sparse Matrix times Sparse Matrix operation
 * \details \f$ C \leftarrow A \ast B \f$ Step 3
 * \param[in] csrRowPtrC_d  Device memory of size rowPtrCtSizeInBytes
 * \param[in] csrRowPtrCt_d Device memory of size rowPtrCtSizeInBytes
 * \param[out] csrColIndCt_d - device memory column indices of temporary CSR output matrix (nnzCt * sizeof(int))
 * \param[out] csrValCt_d  device memory of values of temporary CSR output matrix (nnzCt * sizeof(float/double))
 * \param[out] nnzC number of nonzeroes in output CSR C matrix
 * \param[out] nnzCt - stores nnz of Ct
 * \param[in] spgemmInfo clSparseSpGEMM auxiliary data structre created with clsparseCreateInitSpmSpm
 * \param[in] control A valid clsparseControl created with clsparseCreateControl
 * \ingroup BLAS-3
 */
 CLSPARSE_EXPORT clsparseStatus
     clsparseSpMSpM_ScsrSpmSpmnnzC(cl_mem csrRowPtrC_d,          // ( m+1) * sizeof(int) - memory allocated
                                   cl_mem csrRowPtrCt_d,         // ( m+1) * sizeof(int) - memory allocated
                                   cl_mem* csrColIndCt_d,        // nnzCt * sizeof(int)
                                   cl_mem* csrValCt_d,           // nnzCt * sizeof(float) - Single Precision
                                   int* nnzC,                    // nnz of Output matrix
                                   clSparseSpGEMM spgemmInfo,
                                   const clsparseControl control);


 /*!
 * \brief Fill sparsity pattern and values of output CSR C matrix  (CSR Sparse Matrix times Sparse Matrix operation)
 * \details \f$ C \leftarrow A \ast B \f$ Step 4
 * \param[out] csrRowPtrC_d  row offsets of output C matrix (CSR format)
 * \param[out] csrColIndC_d  Column Indices of output C matrix (CSR format)
 * \param[out] csrValC_d - Values of output C matrix (CSR format) 
 * \param[in] csrRowPtrCt_d  temporary matrix row offsets
 * \param[in] csrColIndCt_d temporary matrix column indices
 * \param[in] csrValCt_d - temporary matrix values 
 * \param[in] nnzC nnz of output C matrix 
 * \param[in] spgemmInfo clSparseSpGEMM auxiliary data structre created with clsparseCreateInitSpmSpm
 * \param[in] control A valid clsparseControl created with clsparseCreateControl
 * \ingroup BLAS-3
 */
 CLSPARSE_EXPORT clsparseStatus
     clsparseSpMSpM_FillScsrOutput(cl_mem csrRowPtrC_d,
                                   cl_mem csrColIndC_d,
                                   cl_mem csrValC_d,
                                   cl_mem csrRowPtrCt_d,
                                   cl_mem csrColIndCt_d,
                                   cl_mem csrValCt_d,
                                   int nnzC, clSparseSpGEMM spgemmInfo, const clsparseControl control);

#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_SPARSE_H_
