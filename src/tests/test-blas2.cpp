/* ************************************************************************
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

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"

//boost ublas
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/blas.hpp>


clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;
//cl_uint ClSparseEnvironment::N = 1024;


namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;


template <typename T>
class Blas2 : public ::testing::Test
{

    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

public:


    void SetUp()
    {
        clsparseInitScalar(&gAlpha);
        clsparseInitScalar(&gBeta);

        clsparseInitVector(&gX);
        clsparseInitVector(&gY);

        hAlpha = T(CSRE::alpha);
        hBeta = T(CSRE::beta);

        hX = uBLAS::vector<T>(CSRE::n_cols, 1);
        hY = uBLAS::vector<T>(CSRE::n_rows, 2);

        cl_int status;

        gX.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   hX.size() * sizeof(T), hX.data().begin(),
                                   &status);
        gX.num_values = hX.size();
        ASSERT_EQ(CL_SUCCESS, status);

        gY.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                   hY.size() * sizeof(T), hY.data().begin(),
                                   &status);
        gY.num_values = hY.size();
        ASSERT_EQ(CL_SUCCESS, status);

        gAlpha.value = clCreateBuffer(CLSE::context,
                                      CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
                                      sizeof(T), &hAlpha, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        gBeta.value = clCreateBuffer(CLSE::context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(T), &hBeta, &status);
        ASSERT_EQ(CL_SUCCESS, status);
    }

    void TearDown()
    {
        ::clReleaseMemObject(gAlpha.value);
        ::clReleaseMemObject(gBeta.value);

        ::clReleaseMemObject(gX.values);
        ::clReleaseMemObject(gY.values);

        clsparseInitScalar(&gAlpha);
        clsparseInitScalar(&gBeta);

        clsparseInitVector(&gX);
        clsparseInitVector(&gY);

    }

    void test_csrmv()
    {
        clsparseStatus status;
        cl_int cl_status;

        if (typeid(T) == typeid(cl_float) )
        {
            status = clsparseScsrmv(&gAlpha, &CSRE::csrSMatrix, &gX,
                                    &gBeta, &gY, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            float* vals = (float*)&CSRE::ublasSCsr.value_data()[0];
            int* rows = &CSRE::ublasSCsr.index1_data()[0];
            int* cols = &CSRE::ublasSCsr.index2_data()[0];
            for (int row = 0; row < CSRE::n_rows; row++)
            {
                hY[row] *= hBeta;
                int row_end = rows[row+1];
                for (int i = rows[row]; i < rows[row+1]; i++)
                    hY[row] += hAlpha * vals[i] * hX[cols[i]];
            }


            T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                                       CL_TRUE, CL_MAP_READ,
                                                       0, gY.num_values * sizeof(T),
                                                       0, nullptr, nullptr, &cl_status);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < hY.size(); i++)
                ASSERT_NEAR(hY[i], host_result[i], fabs(hY[i]*1e-3));

            cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                                  host_result, 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);
        }

        if (typeid(T) == typeid(cl_double) )
        {
            status = clsparseDcsrmv(&gAlpha, &CSRE::csrDMatrix, &gX,
                                    &gBeta, &gY, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            double* vals = (double*)&CSRE::ublasDCsr.value_data()[0];
            int* rows = &CSRE::ublasDCsr.index1_data()[0];
            int* cols = &CSRE::ublasDCsr.index2_data()[0];
            for (int row = 0; row < CSRE::n_rows; row++)
            {
                hY[row] *= hBeta;
                int row_end = rows[row+1];
                for (int i = rows[row]; i < rows[row+1]; i++)
                    hY[row] += hAlpha * vals[i] * hX[cols[i]];
            }

            T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                                       CL_TRUE, CL_MAP_READ,
                                                       0, gY.num_values * sizeof(T),
                                                       0, nullptr, nullptr, &cl_status);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < hY.size(); i++)
                ASSERT_NEAR(hY[i], host_result[i], fabs(hY[i]*1e-10));

            cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                                  host_result, 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);
        }

    }

    uBLAS::vector<T> hX;
    uBLAS::vector<T> hY;

    cldenseVector gX;
    cldenseVector gY;

    T hAlpha;
    T hBeta;

    clsparseScalar gAlpha;
    clsparseScalar gBeta;

};

// CSRMV Tests might give you a false ASSERT_NEAR depends on the matrix.
// We should calculate relative error instead of ASSERT_NEAR.

typedef ::testing::Types<cl_float, cl_double> TYPES;
TYPED_TEST_CASE(Blas2, TYPES);

TYPED_TEST(Blas2, csrmv_adaptive)
{
    if ( typeid(TypeParam) == typeid(cl_float) )
        this->test_csrmv();
    else
    {
//        std::cerr << "Adaptive version of csrmv might crash!" << std::endl;
        this->test_csrmv();
    }
}

TYPED_TEST(Blas2, csrmv_vector)
{
    // To call csrmv vector we need to artificially get rid of the rowBlocks data
    using CSRE = CSREnvironment;

    cl_int cl_status;
    cl_status = clReleaseMemObject(CSRE::csrSMatrix.rowBlocks);
    ASSERT_EQ(CL_SUCCESS, cl_status);
    CSRE::csrSMatrix.rowBlocks = nullptr;

    cl_status = clReleaseMemObject(CSRE::csrDMatrix.rowBlocks);
    ASSERT_EQ(CL_SUCCESS, cl_status);
    CSRE::csrDMatrix.rowBlocks = nullptr;

    CSRE::csrSMatrix.rowBlockSize = 0;
    CSRE::csrDMatrix.rowBlockSize = 0;

    this->test_csrmv();

    // After calling the kernel we need to recreate the rowBlocks data for
    // later use.

    clsparseStatus status;
    status = clsparseCsrMetaSize( &CSRE::csrSMatrix, CLSE::control );
    ASSERT_EQ(clsparseSuccess, status);

    status = clsparseCsrMetaSize( &CSRE::csrDMatrix, CLSE::control );
    ASSERT_EQ(clsparseSuccess, status);

    CSRE::csrSMatrix.rowBlocks =
            ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                              CSRE::csrSMatrix.rowBlockSize * sizeof( cl_ulong ),
                              NULL, &cl_status );

    ASSERT_EQ(CL_SUCCESS, cl_status);

    CSRE::csrDMatrix.rowBlocks = CSRE::csrSMatrix.rowBlocks;
    ::clRetainMemObject( CSRE::csrDMatrix.rowBlocks );

    status = clsparseCsrMetaCompute(&CSRE::csrSMatrix, CLSE::control );
    ASSERT_EQ (clsparseSuccess, status);
    status = clsparseCsrMetaCompute(&CSRE::csrDMatrix, CLSE::control );
    ASSERT_EQ (clsparseSuccess, status);
}


int main (int argc, char* argv[])
{

    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

    std::string path;
    double alpha;
    double beta;
    std::string platform;
    cl_platform_type pID;
    cl_uint dID;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("path,p", po::value(&path)->required(), "Path to matrix in mtx format.")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&dID)->default_value(0),
             "Device id within platform.")
            ("alpha,a", po::value(&alpha)->default_value(1.0),
             "Alpha parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("beta,b", po::value(&beta)->default_value(1.0),
             "Beta parameter for eq: \n\ty = alpha * M * x + beta * y");


    po::variables_map vm;
    po::parsed_options parsed =
            po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );

    try {
        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    std::vector< std::string > to_pass_further =
            po::collect_unrecognized( parsed.options, po::include_positional );

    //check platform
    if(vm.count("platform"))
    {
        if ("AMD" == platform)
        {
            pID = AMD;
        }
        else if ("NVIDIA" == platform)
        {
            pID = NVIDIA;
        }
        else
        {

            std::cout << "The platform option is missing or is ill defined!\n";
            std::cout << "Given [" << platform << "]" << std::endl;
            platform = "AMD";
            pID = AMD;
            std::cout << "Setting [" << platform << "] as default" << std::endl;
        }

    }

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE(pID, dID));
    ::testing::AddGlobalTestEnvironment( new CSRE(path, alpha, beta,
                                                  CLSE::queue, CLSE::context));

    return RUN_ALL_TESTS();

}
