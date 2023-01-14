#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <CL/cl.h>

const char* getOpenCLErrorString(cl_int error);
void checkOpenCLError(cl_int error);

// display device info.
template <typename T>
void appendBitfield(T, T, const std::string&, std::string&);

template <typename T>
class PlatformInfo {
public:
    static void display(cl_platform_id platform, cl_platform_info param_name, std::string str);
};

template <typename T>
class DeviceInfo {
public:
    static void display(cl_device_id device, cl_device_info param_name, std::string str);
};

int main()
{
    // Initialize error code, will use it throughout the OpenCL program.
    cl_int err;

    // cl_uint -> Get the number of platforms supported by the computer system.
    cl_uint numPlatforms;
    checkOpenCLError(clGetPlatformIDs(0, nullptr, &numPlatforms));
    std::cout << "# supported OpenCL platforms: " << numPlatforms << std::endl;
    // Store the structure platform IDs of supported platforms.
    std::vector<cl_platform_id> platforms(numPlatforms);
    checkOpenCLError(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));
    // print out the detailed information for each platform. And select he Nvidia CUDA platform as target OpenCL platform.
    cl_platform_id platform;
    for(size_t idx = 0; idx < numPlatforms; ++idx) {
        size_t size;

        checkOpenCLError(clGetPlatformInfo(platforms[idx],CL_PLATFORM_NAME,0, nullptr,&size));
        char* platformName = new char[size];
        checkOpenCLError(clGetPlatformInfo(platforms[idx],CL_PLATFORM_NAME,size, platformName, nullptr));

        checkOpenCLError(clGetPlatformInfo(platforms[idx],CL_PLATFORM_VENDOR,0, nullptr,&size));
        char* vendorName = new char[size];
        checkOpenCLError(clGetPlatformInfo(platforms[idx],CL_PLATFORM_VENDOR,size, vendorName, nullptr));

        checkOpenCLError(clGetPlatformInfo(platforms[idx],CL_PLATFORM_VERSION,0, nullptr,&size));
        char* versionName = new char[size];
        checkOpenCLError(clGetPlatformInfo(platforms[idx],CL_PLATFORM_VERSION,size, versionName, nullptr));

        std::cout << "Platform name: " << platformName << " --- " << "Vendor name : " << vendorName << " --- " << "Supported version: " << versionName << std::endl;

        if(std::string(platformName).find("NVIDIA") != std::string::npos) {
            platform = platforms[idx];
        }
    }
    std::cout << "...\n";

    // cl_uint -> Get the NVIDIA GPU devices supported by the NVIDIA OpenCL platform. There is only 1 Nvidia GPU device, so directly get it.
    cl_device_id device;
    checkOpenCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    // print out the detailed information for the device.
    DeviceInfo<cl_device_type>::display(device, CL_DEVICE_TYPE, "Device Type");
    DeviceInfo<char>::display(device, CL_DEVICE_VENDOR, "Device Vendor");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_VENDOR_ID, "Device Vendor ID");
    DeviceInfo<char>::display(device, CL_DEVICE_NAME, "Device Name");
    DeviceInfo<char>::display(device, CL_DRIVER_VERSION, "Device Driver Version");
    DeviceInfo<char>::display(device, CL_DEVICE_VERSION, "Device Supported OpenCL Version");
    DeviceInfo<char>::display(device, CL_DEVICE_PROFILE, "Device Supported OpenCL Profile");
    DeviceInfo<char>::display(device, CL_DEVICE_OPENCL_C_VERSION, "Device Compiler Supported OpenCL C Version");
    DeviceInfo<char>::display(device, CL_DEVICE_EXTENSIONS, "Device Supported OpenCL Extensions");
    std::cout << "...\n";

    DeviceInfo<cl_platform_id>::display(device, CL_DEVICE_PLATFORM, "Device Associated OpenCL Platform Name");
    DeviceInfo<cl_bool>::display(device, CL_DEVICE_AVAILABLE, "Device Command Queues Supported");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_ADDRESS_BITS, "Device Address Space Bits");
    DeviceInfo<size_t>::display(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, "Device Timer Resolution");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_REFERENCE_COUNT, "Device Reference Count");
    DeviceInfo<cl_bool>::display(device, CL_DEVICE_ENDIAN_LITTLE, "Device Endian Little");
    DeviceInfo<cl_bool>::display(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, "Device Error Correction Support");
    std::cout << "...\n";

    DeviceInfo<cl_bool>::display(device, CL_DEVICE_COMPILER_AVAILABLE, "Device Program Compiler Exist");
    DeviceInfo<cl_bool>::display(device, CL_DEVICE_LINKER_AVAILABLE, "Device Program Linker Exist");
    DeviceInfo<cl_device_exec_capabilities>::display(device, CL_DEVICE_EXECUTION_CAPABILITIES, "Device Kernel Type Supported");
    DeviceInfo<char>::display(device, CL_DEVICE_BUILT_IN_KERNELS, "Device Supported Built-in Kernels");
    std::cout << "...\n";

    DeviceInfo<cl_device_fp_config>::display(device, CL_DEVICE_SINGLE_FP_CONFIG, "Device Single FP Capabilities");
    DeviceInfo<cl_device_fp_config>::display(device, CL_DEVICE_DOUBLE_FP_CONFIG, "Device Double FP Capabilities");
    std::cout << "...\n";

    DeviceInfo<cl_uint>::display(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "Device Max Clock Frequency");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_MAX_COMPUTE_UNITS, "Device Compute Units Count");
    DeviceInfo<size_t>::display(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "Device Work-Item Global/Local ID Dimension Rank");
    DeviceInfo<size_t>::display(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, "Device Work-Item Max Number in Work-Group");
    DeviceInfo<size_t>::display(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, "Device Work-Item Max Number in Work-Group in Each Dimension");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_MAX_NUM_SUB_GROUPS, "Device Max Sub-Groups Number in Work-Group");

    std::cout << "...\n";

    DeviceInfo<cl_uint>::display(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "Device Sub-Buffer Offset Required Alignment");
    DeviceInfo<cl_ulong>::display(device, CL_DEVICE_GLOBAL_MEM_SIZE, "Device Global Memory Size");
    DeviceInfo<cl_ulong>::display(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, "Device Max Global Memory Allocation");
    DeviceInfo<size_t>::display(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, "Device Single Global Variable Max Storage");
    DeviceInfo<size_t>::display(device, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, "Device All Program Variables Maximum Preferred Global Memory Total Size");
    std::cout << "...\n";

    DeviceInfo<cl_device_mem_cache_type>::display(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, "Device Global L2-Cache Type");
    DeviceInfo<cl_ulong>::display(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "Device Global L2-Cache Size");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "Device Global L2-Cache-Line Size");
    std::cout << "...\n";

    DeviceInfo<cl_device_local_mem_type>::display(device, CL_DEVICE_LOCAL_MEM_TYPE, "Device Local Memory Type");
    DeviceInfo<cl_ulong>::display(device, CL_DEVICE_LOCAL_MEM_SIZE, "Device Local Memory Size");
    std::cout << "...\n";

    DeviceInfo<cl_ulong>::display(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "Device Single Constant Buffer Max Allocation");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_MAX_CONSTANT_ARGS, "Device Constant Arguments Max Supported Count");
    std::cout << "...\n";

    DeviceInfo<size_t>::display(device, CL_DEVICE_MAX_PARAMETER_SIZE, "Device Single Kernel Parameters Max Total Memory Size");
    DeviceInfo<size_t>::display(device, CL_DEVICE_PRINTF_BUFFER_SIZE, "Device Kernel Internal Printf Buffer Size");
    std::cout << "...\n";

    // What we normally use is the host command queue. Nvidia seems not support device command queue.
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES, "Device Queue Max Count per OpenCL Context");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_MAX_ON_DEVICE_EVENTS, "Device Queue Max Event Count");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, "Device Queue Max Size");
    DeviceInfo<cl_uint>::display(device, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, "Device Queue Preferred Size");
    DeviceInfo<cl_command_queue_properties>::display(device, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, "Device Queue Properties");
    DeviceInfo<cl_command_queue_properties>::display(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, "Host Command Queue Properties");
    std::cout << "...\n";

    DeviceInfo<cl_bool>::display(device, CL_DEVICE_IMAGE_SUPPORT, "Device Image Memory Object Support");

    return 0;
}

const char* getOpenCLErrorString(cl_int error) {
    switch(error) {
        // run-time and JIT compiler errors
        case 0:
            return "CL_SUCCESS";
        case -1:
            return "CL_DEVICE_NOT_FOUND";
        case -2:
            return "CL_DEVICE_NOT_AVAILABLE";
        case -3:
            return "CL_COMPILER_NOT_AVAILABLE";
        case -4:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5:
            return "CL_OUT_OF_RESOURCES";
        case -6:
            return "CL_OUT_OF_HOST_MEMORY";
        case -7:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8:
            return "CL_MEM_COPY_OVERLAP";
        case -9:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case -10:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11:
            return "CL_BUILD_PROGRAM_FAILURE";
        case -12:
            return "CL_MAP_FAILURE";
        case -13:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case -16:
            return "CL_LINKER_NOT_AVAILABLE";
        case -17:
            return "CL_LINK_PROGRAM_FAILURE";
        case -18:
            return "CL_DEVICE_PARTITION_FAILED";
        case -19:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30:
            return "CL_INVALID_VALUE";
        case -31:
            return "CL_INVALID_DEVICE_TYPE";
        case -32:
            return "CL_INVALID_PLATFORM";
        case -33:
            return "CL_INVALID_DEVICE";
        case -34:
            return "CL_INVALID_CONTEXT";
        case -35:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case -36:
            return "CL_INVALID_COMMAND_QUEUE";
        case -37:
            return "CL_INVALID_HOST_PTR";
        case -38:
            return "CL_INVALID_MEM_OBJECT";
        case -39:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40:
            return "CL_INVALID_IMAGE_SIZE";
        case -41:
            return "CL_INVALID_SAMPLER";
        case -42:
            return "CL_INVALID_BINARY";
        case -43:
            return "CL_INVALID_BUILD_OPTIONS";
        case -44:
            return "CL_INVALID_PROGRAM";
        case -45:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46:
            return "CL_INVALID_KERNEL_NAME";
        case -47:
            return "CL_INVALID_KERNEL_DEFINITION";
        case -48:
            return "CL_INVALID_KERNEL";
        case -49:
            return "CL_INVALID_ARG_INDEX";
        case -50:
            return "CL_INVALID_ARG_VALUE";
        case -51:
            return "CL_INVALID_ARG_SIZE";
        case -52:
            return "CL_INVALID_KERNEL_ARGS";
        case -53:
            return "CL_INVALID_WORK_DIMENSION";
        case -54:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case -55:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case -56:
            return "CL_INVALID_GLOBAL_OFFSET";
        case -57:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case -58:
            return "CL_INVALID_EVENT";
        case -59:
            return "CL_INVALID_OPERATION";
        case -60:
            return "CL_INVALID_GL_OBJECT";
        case -61:
            return "CL_INVALID_BUFFER_SIZE";
        case -62:
            return "CL_INVALID_MIP_LEVEL";
        case -63:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64:
            return "CL_INVALID_PROPERTY";
        case -65:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66:
            return "CL_INVALID_COMPILER_OPTIONS";
        case -67:
            return "CL_INVALID_LINKER_OPTIONS";
        case -68:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        // extension errors
        case -1000:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001:
            return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002:
            return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003:
            return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004:
            return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005:
            return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default:
            return "Unknown OpenCL error";
    }
}

void checkOpenCLError(cl_int error) {
    if(error != CL_SUCCESS) {
        fprintf(stderr, "%s @ %d: %s", __FILE__, __LINE__, getOpenCLErrorString(error));

        exit(1);
    }
}

template <typename T>
void PlatformInfo<T>::display(cl_platform_id platform, cl_platform_info param_name, std::string str) {
    size_t size;
    checkOpenCLError( clGetPlatformInfo(platform, param_name, 0, nullptr, &size) );
    auto info = new T[size];
    checkOpenCLError( clGetPlatformInfo(platform, param_name, size, info, nullptr) );

    switch(param_name) {
        case CL_PLATFORM_NAME: {
            auto platformName = reinterpret_cast<char*>(info);
            std::cout << str << " : " << platformName << std::endl;
        }
            break;
        case CL_PLATFORM_VENDOR: {
            auto platformVendor = reinterpret_cast<char*>(info);
            std::cout << str << " : " << platformVendor << std::endl;
        }
            break;
        case CL_PLATFORM_VERSION: {
            // OpenCL version string. Returns the OpenCL version supported by the platform. OpenCL<space><major_version.minor_version><space><vendor-specific information>
            auto platformSupportedOpenCLVersion = reinterpret_cast<char*>(info);
            std::cout << str << " : " << platformSupportedOpenCLVersion << std::endl;
        }
            break;
        case CL_PLATFORM_PROFILE: {
            // OpenCL profile string. Returns the profile name supported by the device (see note). The profile name returned can be one of the following strings:
            // FULL_PROFILE - if the device supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported).
            // EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile.
            // The platform profile returns the profile that is implemented by the OpenCL framework.
            // If the platform profile returned is FULL_PROFILE, the OpenCL framework will support devices that are FULL_PROFILE and may also support devices that are EMBEDDED_PROFILE.
            // The compiler must be available for all devices i.e. CL_DEVICE_COMPILER_AVAILABLE is CL_TRUE.
            // If the platform profile returned is EMBEDDED_PROFILE, then devices that are only EMBEDDED_PROFILE are supported.
            auto platformSupportedOpenCLProfile = reinterpret_cast<char*>(info);
            std::cout << str << " : " << platformSupportedOpenCLProfile << std::endl;
        }
            break;
        case CL_PLATFORM_EXTENSIONS: {
            // Returns a space separated list of extension names (the extension names themselves do not contain any spaces) supported by the device.
            // The following approved Khronos extension names must be returned by all device that support OpenCL C 2.0:
            //      cl_khr_byte_addressable_store
            //      cl_khr_fp64 (for backward compatibility if double precision is supported)
            //      cl_khr_3d_image_writes
            //      cl_khr_image2d_from_buffer
            //      cl_khr_depth_images
            // Please refer to the OpenCL 2.0 Extension Specification for a detailed description of these extensions.
            auto platformSupportedOpenCLExtensions = reinterpret_cast<char*>(info);
            std::cout << str << " : " << platformSupportedOpenCLExtensions << std::endl;
        }
            break;
        // ...
        default:
            break;
    }

    free(info);
}

template <typename T>
void appendBitfield(T info, T value, const std::string& name, std::string& str) {
    if (info & value) {
        if (str.length() > 0) {
            str.append(" : ");
        }
        str.append(name);
    }
}
