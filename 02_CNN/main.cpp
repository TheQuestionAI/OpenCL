#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>

#define CL_TARGET_OPENCL_VERSION 210
#define CL_HPP_TARGET_OPENCL_VERSION 210

#include <CL/cl.h>
#include <CL/opencl.hpp>

// Get the target OpenCL platform by vendor argument
cl::Platform GetTargetPlatform(const std::string& vendor);
// Get the first available specified device_type OpenCL device under given platform.
cl::Device GetTargetDevice(const cl::Platform& platform, const std::string& device_type = "GPU");
// Create a OpenCL context under given platform, and associate a given device to it.
cl::Context CreateContext(const cl::Platform& platform, const cl::Device& device);
// Create a command queue under given context, and associate it to a device within the context.
cl::CommandQueue CreateCommandQueue(const cl::Context& context, const cl::Device& device, const cl::QueueProperties& queueProperties = cl::QueueProperties::Profiling);
// Create a program under given context, and build it with provided build option for given device within the context.
cl::Program CreateAndBuildProgram(const cl::Context& context, const cl::Device& device, const std::string& kernelSourcePath, const std::string& buildOption = {});
// Create a kernel given a build program.
cl::Kernel CreateKernel(const cl::Program& program, const std::string& kernel_name);
// Create a Image2DArray memory object.
cl::Image2DArray CreateImage2DArray(const cl::Context& context, cl_mem_flags flags, const std::vector<size_t>& shape, const std::string& dataSource = { }, const std::vector<size_t>& pitch = {0, 0}, const cl::ImageFormat& format = {CL_RGBA, CL_FLOAT});
// Create a Image2D memory object.
cl::Image2D CreateImage2D(const cl::Context& context, cl_mem_flags flags, const std::vector<size_t>& shape, const std::string& dataSource = { }, const size_t& pitch = 0, const cl::ImageFormat& format = {CL_RGBA, CL_FLOAT});
// Create a Image1D memory object.
cl::Image1D CreateImage1D(const cl::Context& context, cl_mem_flags flags, const std::vector<size_t>& shape, const std::string& dataSource = { }, const cl::ImageFormat& format = {CL_RGBA, CL_FLOAT});

// Set Conv3D kernel arguments.
std::pair<std::vector<std::vector<size_t>>, std::vector<size_t>> ProcessConv3DKernelScalarArguments(const std::string& parameterSource);
void SetConv3DKernelArguments(cl::Kernel& conv3DKernel, cl::Image2DArray& input, cl::Image2D& filters, cl::Image1D& biases, cl::Image2DArray& output, const std::vector<size_t>& scalarArgs);

// Read raw data from Image2DArray memory object.
std::vector<char> ReadRawDataFromImage2DArray(cl::CommandQueue& command_queue, const cl::Image2DArray& src);
std::vector<char> LoadBinaryDataFromFile(const std::string& dataSource);

void printVectorData(const std::vector<char>& rawData, const std::string& targetDataType = "float");
void validationResults(const std::vector<char>& result, const std::vector<char>& groundTruth, const std::string& targetDataType = "float");

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec);

size_t align(size_t src, size_t base) {
    return (src + base - 1) / base * base;
}

int main() {
    std::cout << "\n";

    auto platform = GetTargetPlatform("NVIDIA");
    auto device = GetTargetDevice(platform, "GPU");
    auto context = CreateContext(platform, device);
    auto commandQueue = CreateCommandQueue(context, device);

    const std::string kernel_name = "conv3D";
    const std::string kernelSourcePath = "D:\\Dropbox\\OMSCS_Yangzi\\CS textbook\\Computer Graphics for Video Games\\ClionOpenCL\\OpenCL_Kernel_Test_Framework\\kernel\\conv3D.cl";
    int M = 3;
    const std::string buildOption = "-DMW=" + std::to_string(M) + " -DMH=" + std::to_string(M) + " -DMD=" + std::to_string(M) + " -DMC=" + std::to_string(M);
    auto program = CreateAndBuildProgram(context, device, kernelSourcePath);
    auto kernel = CreateKernel(program, kernel_name);

    const std::string inputSource = "D:\\Dropbox\\OMSCS_Yangzi\\CS textbook\\Computer Graphics for Video Games\\ClionOpenCL\\OpenCL_Kernel_Test_Framework\\data\\input.raw";
    const std::string filtersSource = "D:\\Dropbox\\OMSCS_Yangzi\\CS textbook\\Computer Graphics for Video Games\\ClionOpenCL\\OpenCL_Kernel_Test_Framework\\data\\filters.raw";
    const std::string biasesSource = "D:\\Dropbox\\OMSCS_Yangzi\\CS textbook\\Computer Graphics for Video Games\\ClionOpenCL\\OpenCL_Kernel_Test_Framework\\data\\biases.raw";
    const std::string tfOutputSource = "D:\\Dropbox\\OMSCS_Yangzi\\CS textbook\\Computer Graphics for Video Games\\ClionOpenCL\\OpenCL_Kernel_Test_Framework\\data\\tf_output.raw";
    const std::string paramSource = "D:\\Dropbox\\OMSCS_Yangzi\\CS textbook\\Computer Graphics for Video Games\\ClionOpenCL\\OpenCL_Kernel_Test_Framework\\data\\parameters.raw";
    //SetConv3DKernelArguments(context, kernel, inputSource, filtersSource, paramSource);
    auto paramPair = ProcessConv3DKernelScalarArguments(paramSource);
    auto imgShapes = paramPair.first;
    auto scalarArguments = paramPair.second;

    std::cout << scalarArguments << std::endl;

    // input image2darray: Cgin x Din x Hin x Win*4cin -> Cgin*Din x Hin x Win
    auto input = CreateImage2DArray(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, imgShapes[0], inputSource);
    // filters image2d: Cgin x Dk x Hk x Wk x Cout*4cin -> Cgin*Dk*Hk*Wk x Cout
    auto filters = CreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, imgShapes[1], filtersSource);
    // biases image1d: Cgout*4cout -> Cout
    auto biases = CreateImage1D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, imgShapes[2], biasesSource);
    // // output image2darray: Cgout x Dout x Hout x Wout*4cout -> Cgout*Dout x Hout x Wout
    auto output = CreateImage2DArray(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, imgShapes[3]);

    std::cout << "\nInput Image2DArray Shape: " << imgShapes[0] << std::endl;
    std::cout << "Filters Image2D Shape: " << imgShapes[1] << std::endl;
    std::cout << "Biases Image1D Shape: " << imgShapes[2] << std::endl;
    std::cout << "Output Image2DArray Shape: " << imgShapes[3] << std::endl;
    std::cout << "Scalar Arguments: " << scalarArguments << std::endl;
    // Set kernel arguments.
    SetConv3DKernelArguments(kernel, input, filters, biases, output, scalarArguments);


    auto Wout = scalarArguments[2];
    auto Hout = scalarArguments[3];
    auto Dout = scalarArguments[4];
    auto Cgout = scalarArguments[5];
    auto Cout = Cgout * 4;
    std::cout << "Original Host Output Shape: " << std::vector<size_t>({Wout, Hout, Dout, Cout}) << std::endl;

    // create kernel global_work_size and local_work_size;
    /*
    cl::NDRange globalWorkSize = {align(align(Wout, M) / M, 4),
                                  align(align(Hout, M) / M, 4),
                                  align(align(Dout * Cout, M * M) / (M * M), 4)};
    */
    cl::NDRange globalWorkSize = {align(Wout, 32),
                                  align(Hout, 4),
                                  align(Dout, 4)};
    cl::NDRange localWorkSize = {32, 4, 4};

    // pass the kernel to command_queue for execution.
    cl_int err;
    cl::Event prof_event;
    err = commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, localWorkSize, nullptr, &prof_event);
    commandQueue.finish();
    prof_event.wait();

    if(err != CL_SUCCESS) {
        std::cerr << "Failed to execute the kernel...." << std::endl;
        exit(1);
    }

    auto rawOutput = ReadRawDataFromImage2DArray(commandQueue, output);
    auto tfOutput = LoadBinaryDataFromFile(tfOutputSource);
    validationResults(rawOutput, tfOutput);
    //printVectorData(rawOutput, "float");

    /*
    std::cout << '\n';
    size_t maxWorkGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    std::cout << "Maximum work group size: " << maxWorkGroupSize << std::endl;
    */

    /*
    size_t maxSubGroupSize = 0;
    kernel.getSubGroupInfo(device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,localWorkSize, &maxSubGroupSize);
    // auto maxSubGroupSize = kernel.getSubGroupInfo<CL_KERNEL_MAX_NUM_SUB_GROUPS>(device, localWorkSize);
    std::cout << "Maximum wave size: " << maxSubGroupSize << std::endl;
    */

    return 0;
}

cl::Platform GetTargetPlatform(const std::string& vendor) {
    cl_int err;
    // Get list of all available opencl platform.
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    if(err != CL_SUCCESS || platforms.empty()) {
        std::cerr << "There are no available OpenCL platforms!" << std::endl;
        exit(1);
    }

    // Find the target OpenCL platform specified by vendor argument.
    cl::Platform targetPlatform;
    for(const auto& platform : platforms) {
        // platformName是std::string类型.
        const auto platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if(platformName.find(vendor) != std::string::npos) {
            targetPlatform = platform;
            break;
        }
    }

    // platformName是std::string类型. 如果targetPlatform未被初始化, 即未找到platform name, 则platformName是一个empty string.
    const auto targetPlatformName = targetPlatform.getInfo<CL_PLATFORM_NAME>();
    if(targetPlatformName.empty()) {
        std::cerr << "There is no " << vendor << " OpenCL platform available!" << std::endl;
        exit(1);
    }

    std::cout << "Successfully get the " << vendor << " OpenCL platform......" << std::endl;

    return targetPlatform;
}

cl::Device GetTargetDevice(const cl::Platform& platform, const std::string& device_type) {
    if(device_type != "CPU" && device_type != "GPU") {
        std::cerr << "Invalid device_type: Must be 'GPU' or 'CPU'." << std::endl;
        exit(1);
    }
    std::unordered_map<std::string, cl_device_type> table = {{"GPU", CL_DEVICE_TYPE_GPU}, {"CPU", CL_DEVICE_TYPE_CPU}};

    cl_int err;
    // Get list of all available OpenCL device_type devices given target platform.
    std::vector<cl::Device> devices;
    err = platform.getDevices(table[device_type], &devices);
    if(err != CL_SUCCESS || devices.empty()) {
        std::cerr << "There are no available OpenCL devices under platform '" << platform.getInfo<CL_PLATFORM_NAME>() << "'." << std::endl;
        exit(1);
    }

    // Use the first available device as target device.
    cl::Device targetDevice = devices[0];
    std::cout << "Successfully get the target OpenCL device......" << std::endl;
    std::cout << "Target device type: " << device_type << std::endl;
    std::cout << "Target device name: " << targetDevice.getInfo<CL_DEVICE_NAME>() << std::endl;

    return targetDevice;
}

cl::Context CreateContext(const cl::Platform& platform, const cl::Device& device) {
    cl_int err;
    // create OpenCL context, bind target platform and target device.
    // Platform class overlaod operator(), so platform() directly returns the underlying cl_platform_id.
    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
    cl::Context context(device, context_properties, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
    }

    std::cout << "Successfully creat a OpenCL context given target platform and device......" << std::endl;

    return context;
}

cl::CommandQueue CreateCommandQueue(const cl::Context& context, const cl::Device& device, const cl::QueueProperties& queueProperties) {
    cl_int err;
    // Create OpenCL command_queue, bind target context and target device.
    cl::CommandQueue commandQueue(context, device, queueProperties, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL command_queue." << std::endl;
        exit(1);
    }

    std::cout << "Successfully create a command_queue given target context and device......" << std::endl;

    return commandQueue;
}

std::string LoadKernelSource(const std::string& sourceFilePath) {
    std::ifstream file(sourceFilePath);
    if(!file.is_open()) {
        std::cerr << "Failed to open: " << sourceFilePath << std::endl;
        exit(1);
    }

    std::string source;
    std::string line;
    while(std::getline(file, line)) {
        source += line + '\n';
    }

    return source;
}

cl::Program CreateAndBuildProgram(const cl::Context& context, const cl::Device& device, const std::string& kernelSourcePath, const std::string& buildOption) {
    cl_int err;

    const auto& kernelSource = LoadKernelSource(kernelSourcePath);
    cl::Program program(context, kernelSource, false, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        exit(1);
    }

    std::cout << "Successfully create a program with provided kernel source under given context......" << std::endl;

    err = program.build(device, buildOption.c_str());
    if(err != CL_SUCCESS) {
        std::cerr << "OpenCL compilation error\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    std::cout << "Successfully build a program under given device......" << std::endl;
    std::cout << "Build options: " << buildOption << std::endl;

    return program;
}

cl::Kernel CreateKernel(const cl::Program& program, const std::string& kernel_name) {
    cl_int err;

    // extract kernel from built program.
    cl::Kernel kernel(program, kernel_name.c_str(), &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL kernel:" << kernel_name << std::endl;
    }

    std::cout << "Successfully create a kernel given target built program......" << std::endl;
    std::cout << "Kernel name: " << kernel_name << std::endl;

    return kernel;
}

std::vector<char> LoadBinaryDataFromFile(const std::string& dataSource) {
    std::ifstream file(dataSource, std::ios::binary | std::ios::in | std::ios::ate);
    if(!file.is_open()) {
        std::cerr << "Failed to open: " << dataSource << std::endl;
        exit(1);
    }

    long long size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buffer = new char[size];
    file.read(buffer, size);

    std::vector<char> data(buffer, buffer + size);
    delete[] buffer;

    return data;
}

std::pair<void*, size_t> LoadRawBinaryDataFromFile(const std::string& dataSource) {
    std::ifstream file(dataSource, std::ios::binary | std::ios::in | std::ios::ate);
    if(!file.is_open()) {
        std::cerr << "Failed to open: " << dataSource << std::endl;
        exit(1);
    }

    long long size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buffer = new char[size];
    file.read(buffer, size);

    return {buffer, size};
}

cl::Image2DArray CreateImage2DArray(const cl::Context& context, cl_mem_flags flags, const std::vector<size_t>& shape, const std::string& dataSource, const std::vector<size_t>& pitch, const cl::ImageFormat& format) {
    cl_int err;

    void* hostPtr;
    if(dataSource.empty()) {
        hostPtr = nullptr;
    }
    else if(dataSource == "random") {
        hostPtr = nullptr;      // implement random values later.
    }
    else {
        auto rawDataStruct = LoadRawBinaryDataFromFile(dataSource);
        hostPtr = rawDataStruct.first;
        }

    size_t width = shape[0];
    size_t height = shape[1];
    size_t arraySize = shape[2];
    size_t rowPitch = pitch[0];
    size_t slicePitch = pitch[1];

    cl::Image2DArray object(context, flags, format, arraySize, width, height, rowPitch, slicePitch, hostPtr, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL Image2DArray memory object." << std::endl;
        exit(1);
    }

    std::cout << "Successfully create a OpenCL Image2DArray memory object given a context......" << std::endl;

    delete (char*)hostPtr;

    return object;
}

cl::Image2D CreateImage2D(const cl::Context& context, cl_mem_flags flags, const std::vector<size_t>& shape, const std::string& dataSource, const size_t& pitch, const cl::ImageFormat& format) {
    cl_int err;

    void* hostPtr;
    if(dataSource.empty()) {
        hostPtr = nullptr;
    }
    else if(dataSource == "random") {
        hostPtr = nullptr;      // implement random values later.
    }
    else {
        auto rawDataStruct = LoadRawBinaryDataFromFile(dataSource);
        hostPtr = rawDataStruct.first;
    }

    size_t width = shape[0];
    size_t height = shape[1];

    cl::Image2D object(context, flags, format, width, height, pitch, hostPtr, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL Image2D memory object." << std::endl;
        exit(1);
    }

    std::cout << "Successfully create a OpenCL Image2D memory object given a context......" << std::endl;

    delete (char*)hostPtr;

    return object;
}

cl::Image1D CreateImage1D(const cl::Context& context, cl_mem_flags flags, const std::vector<size_t>& shape, const std::string& dataSource, const cl::ImageFormat& format) {
    cl_int err;

    void* hostPtr;
    if(dataSource.empty()) {
        hostPtr = nullptr;
    }
    else if(dataSource == "random") {
        hostPtr = nullptr;      // implement random values later.
    }
    else {
        auto rawDataStruct = LoadRawBinaryDataFromFile(dataSource);
        hostPtr = rawDataStruct.first;
    }

    size_t width = shape[0];

    cl::Image1D object(context, flags, format, width, hostPtr, &err);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL Image2D memory object." << std::endl;
        exit(1);
    }

    std::cout << "Successfully create a OpenCL Image2D memory object given a context......" << std::endl;

    delete (char*)hostPtr;

    return object;
}


std::pair<std::vector<std::vector<size_t>>, std::vector<size_t>> ProcessConv3DKernelScalarArguments(const std::string& parameterSource) {
    std::vector<char> rawParamData = LoadBinaryDataFromFile(parameterSource);
    std::vector<int> parameter(reinterpret_cast<int*>(rawParamData.data()), reinterpret_cast<int*>(rawParamData.data()) + rawParamData.size() / sizeof(int));

    std::vector<std::vector<size_t>> img_shapes;
    int Din = parameter[0];
    int Hin = parameter[1];
    int Win = parameter[2];
    int Cin = parameter[3];

    int Dout = parameter[4];
    int Hout = parameter[5];
    int Wout = parameter[6];
    int Cout = parameter[7];

    int Dk = parameter[8];
    int Hk = parameter[9];
    int Wk = parameter[10];

    int Sx = parameter[11];
    int Sy = parameter[12];
    int Sz = parameter[13];

    int Px = parameter[14];
    int Py = parameter[15];
    int Pz = parameter[16];

    int Lx = parameter[17];
    int Ly = parameter[18];
    int Lz = parameter[19];

    // input image2darray: Cgin x Din x Hin x Win*4cin -> Cgin*Din x Hin x Win
    size_t img2darr_input_width = Win;
    size_t img2darr_input_height = Hin;
    size_t img2darr_input_depth = Din * Cin / 4;
    std::vector<size_t> img2darr_input_shape = {img2darr_input_width, img2darr_input_height, img2darr_input_depth};
    img_shapes.push_back(img2darr_input_shape);

    // filters image2d: Cgin x Dk x Hk x Wk x Cout*4cin -> Cgin*Dk*Hk*Wk x Cout
    size_t img_filters_width = Cout;
    size_t img_filters_height = Dk * Hk * Wk * Cin / 4;
    std::vector<size_t> img_filters_shape = {img_filters_width, img_filters_height};
    img_shapes.push_back(img_filters_shape);

    // biases image1d: Cgout*4cout
    size_t img_biases_width = Cout / 4;
    std::vector<size_t> img_biases_shape = {img_biases_width};
    img_shapes.push_back(img_biases_shape);

    // output image2darray: Cgout x Dout x Hout x Wout*4cout -> Cgout*Dout x Hout x Wout
    size_t img2darr_output_width = Wout;
    size_t img2darr_output_height = Hout;
    size_t img2darr_output_depth = Dout * Cout / 4;
    std::vector<size_t> img2darr_output_shape = {img2darr_output_width, img2darr_output_height, img2darr_output_depth};
    img_shapes.push_back(img2darr_output_shape);

    std::vector<size_t> kernel_scalar_args;
    kernel_scalar_args.push_back(Din);
    kernel_scalar_args.push_back(Cin / 4);
    kernel_scalar_args.push_back(Wout);
    kernel_scalar_args.push_back(Hout);
    kernel_scalar_args.push_back(Dout);
    kernel_scalar_args.push_back(Cout / 4);
    kernel_scalar_args.push_back(Wk);
    kernel_scalar_args.push_back(Hk);
    kernel_scalar_args.push_back(Dk);
    kernel_scalar_args.push_back(Sx);
    kernel_scalar_args.push_back(Sy);
    kernel_scalar_args.push_back(Sz);
    kernel_scalar_args.push_back(Px);
    kernel_scalar_args.push_back(Py);
    kernel_scalar_args.push_back(Pz);
    kernel_scalar_args.push_back(Lx);
    kernel_scalar_args.push_back(Ly);
    kernel_scalar_args.push_back(Lz);

    return {img_shapes, kernel_scalar_args};
}

void SetConv3DKernelArguments(cl::Kernel& kernel, cl::Image2DArray& input, cl::Image2D& filters, cl::Image1D& biases, cl::Image2DArray& output, const std::vector<size_t>& scalarArgs) {
    cl_int err;
    // Set kernel arguments.
    err = kernel.setArg(0, input);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to set input argument." << std::endl;
        exit(1);
    }

    err = kernel.setArg(1, filters);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to set filters argument." << std::endl;
        exit(1);
    }

    err = kernel.setArg(2, biases);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to set biases argument." << std::endl;
        exit(1);
    }

    err = kernel.setArg(3, output);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to set output argument." << std::endl;
        exit(1);
    }

    for(int idx = 0; idx < scalarArgs.size(); ++idx) {
        // It is very crucial that the argument data type passes is exactly the same as the kernel parameter declared.
        err = kernel.setArg(idx + 4, (int)scalarArgs[idx]);
        if(err != CL_SUCCESS) {
            std::cerr << "Failed to set argument: " << idx << std::endl;
            exit(1);
        }

    }
}

std::vector<char> ReadRawDataFromImage2DArray(cl::CommandQueue& command_queue, const cl::Image2DArray& src) {
    auto width = src.getImageInfo<CL_IMAGE_WIDTH>();
    auto height = src.getImageInfo<CL_IMAGE_HEIGHT>();
    auto arraySize = src.getImageInfo<CL_IMAGE_ARRAY_SIZE>();
    auto pixelSize = src.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();

    cl_int err;
    auto hostPtr = reinterpret_cast<char*>(command_queue.enqueueMapImage(src, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0}, {width, height, arraySize}, 0, 0, NULL, NULL, &err));
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to map OpenCL Image2DArray memory object." << std::endl;
        exit(1);
    }

    auto totalSize = width * height * arraySize * pixelSize;
    std::vector<char> rawData(totalSize, 0);
    for(int idx = 0; idx < totalSize; ++idx) {
        rawData[idx] = hostPtr[idx];
    }

    err = command_queue.enqueueUnmapMemObject(src, (void*)hostPtr,NULL, NULL);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to unmap OpenCL Image2DArray memory object." << std::endl;
        exit(1);
    }

    return rawData;
}

void printVectorData(const std::vector<char>& rawData, const std::string& targetDataType) {

    if(targetDataType == "float") {
        using dataType = float;
        std::vector<dataType> data((float*)rawData.data(), (float*)rawData.data() + rawData.size() / sizeof(float));

        std::cout << data << std::endl;
    }
}

void validationResults(const std::vector<char>& result, const std::vector<char>& groundTruth, const std::string& targetDataType) {
    if(result.size() != groundTruth.size()) {
        std::cout << "Two vectors are not in same size!" << std::endl;
        exit(1);
    }

    size_t count = 0;
    if(targetDataType == "float") {
        using dataType = float;
        std::vector<dataType> floatResult((dataType*)result.data(), (dataType*)result.data() + result.size() / sizeof(dataType));
        std::vector<dataType> floatGroundTruth((dataType*)groundTruth.data(), (dataType*)groundTruth.data() + groundTruth.size() / sizeof(dataType));

        auto N = floatResult.size();
        for(int idx = 0; idx < N; ++idx) {
            float diff = std::fabs(floatResult[idx] - floatGroundTruth[idx]);
            std::string res = diff < 0.001f ? "correct" : "wrong";
            if(res == "correct") ++count;

            std::cout << "[ " << res << ", " << floatResult[idx] << ", " << floatGroundTruth[idx] << ", " << diff << " ]" << std::endl;
        }

        if(count == floatResult.size())
            std::cout << "floatResult == floatGroundTruth: Yes !" << std::endl;
        else
            std::cout << "floatResult == floatGroundTruth: No !" << std::endl;
    }
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
    out << "[ ";
    for(auto d : vec) {
        std::cout << d << ", ";
    }
    out << "]";

    return out;
}