#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>
#include <numeric>

#include "resources/clsparse_environment.h"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

namespace po = boost::program_options;

TEST (REDUCE, opencl20_float)
{
#if (BUILD_CLVERSION > 120) //this flag should be defined in tests CMakeLists.txt
    using CLSE = ClSparseEnvironment;

    cl_uint size = 2546872;
    std::vector<cl_float> y(size, 1.0f);

    cl_float zero = 0.f;

    clsparseScalar sum;
    clsparseInitScalar(&sum);


    clsparseVector gY;
    clsparseInitVector(&gY);
    gY.n = y.size();

    gY.values = (cl_float*)::clSVMAlloc(CLSE::context, CL_MEM_READ_WRITE, gY.n * sizeof(cl_float), 0);
    ASSERT_NE(gY.values, nullptr);

    sum.value = (cl_float*)::clSVMAlloc(CLSE::context, CL_MEM_READ_WRITE, sizeof(cl_float), 0);
    ASSERT_NE(sum.value, nullptr);

    cl_int status;
    status = cldenseSreduce(&sum, &gY, CLSE::control);

    ASSERT_EQ(clsparseSuccess, status);

    cl_float ref_sum = std::accumulate(y.begin(), y.end(), 0.0);

    cl_float host_sum = 0.0f;

//    clEnqueueReadBuffer(CLSE::queue,
//                        sum.value, 1, 0,
//                        sizeof(cl_float),
//                        &host_sum, 0, NULL, NULL);

    ASSERT_NEAR(ref_sum, host_sum, 5e-8);
#endif
}

TEST (REDUCE, float_simple)
{
    using CLSE = ClSparseEnvironment;

    cl_uint size = 2546872;
    std::vector<cl_float> y(size, 1.0f);

    cl_float zero = 0.f;

    clsparseScalar sum;
    clsparseInitScalar(&sum);


    clsparseVector gY;
    clsparseInitVector(&gY);

    cl_int status;
    gY.values = ::clCreateBuffer(CLSE::context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 y.size() * sizeof(cl_float), y.data(), &status);
    gY.n = y.size();

    ASSERT_EQ(CL_SUCCESS, status);

    sum.value = ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                               sizeof(cl_float), NULL, &status);
    ASSERT_EQ(CL_SUCCESS, status);

    status = cldenseSreduce(&sum, &gY, CLSE::control);

    ASSERT_EQ(clsparseSuccess, status);

    cl_float ref_sum = std::accumulate(y.begin(), y.end(), 0.0);

    cl_float host_sum = 0.0f;

    clEnqueueReadBuffer(CLSE::queue,
                        sum.value, 1, 0,
                        sizeof(cl_float),
                        &host_sum, 0, NULL, NULL);

    ASSERT_NEAR(ref_sum, host_sum, 5e-8);
}

TEST (REDUCE, double_simple)
{
    using CLSE = ClSparseEnvironment;

    cl_uint size = 512*256;
    std::vector<cl_double> y(size, 2.0f);

    cl_double zero = 0.0;

    clsparseScalar sum;
    clsparseInitScalar(&sum);


    clsparseVector gY;
    clsparseInitVector(&gY);

    cl_int status;
    gY.values = ::clCreateBuffer(CLSE::context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 y.size() * sizeof(cl_double), y.data(), &status);
    gY.n = y.size();

    ASSERT_EQ(CL_SUCCESS, status);

    sum.value = ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                               sizeof(cl_double), NULL, &status);

    ASSERT_EQ(CL_SUCCESS, status);

    status = cldenseDreduce(&sum, &gY, CLSE::control);

    ASSERT_EQ(clsparseSuccess, status);

    cl_double ref_sum = std::accumulate(y.begin(), y.end(), 0.0);

    cl_double host_sum = 0.0;

    status = clEnqueueReadBuffer(CLSE::queue,
                        sum.value, CL_TRUE, 0,
                        1 * sizeof(cl_double),
                        &host_sum, 0, NULL, NULL);

    ASSERT_EQ(CL_SUCCESS, status);

    ASSERT_NEAR(ref_sum, host_sum, 5e-8);
}

int main (int argc, char* argv[])
{

    using CLSE = ClSparseEnvironment;


    std::string platform;
    cl_platform_type pID;
    cl_uint dID;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&dID)->default_value(0),
             "Device id within platform.");


    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }


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
   
    return RUN_ALL_TESTS();

}
