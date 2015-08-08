#include "sparse_matrix_environment.h"

CSRSparseEnvironment::sMatrixType CSRSparseEnvironment::ublasSCsr = CSRSparseEnvironment::sMatrixType();

cl_int CSRSparseEnvironment::n_rows = 0;
cl_int CSRSparseEnvironment::n_cols = 0;
cl_int CSRSparseEnvironment::n_vals = 0;

clsparseCsrMatrix CSRSparseEnvironment::csrSMatrix = clsparseCsrMatrix();


std::string CSRSparseEnvironment::file_name = std::string();