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
void DeviceInfo<T>::display(cl_device_id device, cl_device_info param_name, std::string str) {
    size_t size;
    checkOpenCLError( clGetDeviceInfo(device, param_name, 0, nullptr, &size) );
    auto info = new T[size];
    checkOpenCLError( clGetDeviceInfo(device, param_name, size, info, nullptr) );

    switch(param_name) {
        case CL_DEVICE_TYPE: {
            auto deviceType = reinterpret_cast<cl_device_type*>(info);
            appendBitfield<cl_device_type>(*deviceType, CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU", str);
            appendBitfield<cl_device_type>(*deviceType, CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU", str);
            appendBitfield<cl_device_type>(*deviceType, CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", str);
            appendBitfield<cl_device_type>(*deviceType, CL_DEVICE_TYPE_DEFAULT, "CL_DEVICE_TYPE_DEFAULT", str);

            std::cout << str << std::endl;
        }
            break;
        case CL_DEVICE_NAME: {
            auto deviceName = reinterpret_cast<char*>(info);
            std::cout << str << " : " << deviceName << std::endl;
        }
            break;
        case CL_DEVICE_VENDOR: {
            auto deviceVendor = reinterpret_cast<char*>(info);
            std::cout << str << " : " << deviceVendor << std::endl;
        }
            break;
        case CL_DEVICE_VENDOR_ID: {
            auto deviceVendorID = reinterpret_cast<cl_uint*>(info);
            std::cout << str << " : " << *deviceVendorID << std::endl;
        }
            break;
        case CL_DRIVER_VERSION: {
            // OpenCL's software driver version string in the form major_number.minor_number.
            auto deviceDriverVersion = reinterpret_cast<char*>(info);
            std::cout << str << " : " << deviceDriverVersion << std::endl;
        }
            break;
        case CL_DEVICE_VERSION: {
            // OpenCL version string. Returns the OpenCL version supported by the device. OpenCL<space><major_version.minor_version><space><vendor-specific information>
            auto deviceSupportedOpenCLVersion = reinterpret_cast<char*>(info);
            std::cout << str << " : " << deviceSupportedOpenCLVersion << std::endl;
        }
            break;
        case CL_DEVICE_OPENCL_C_VERSION: {
            // OpenCL C version string. Returns the highest OpenCL C version supported by the compiler for this device that is not of type CL_DEVICE_TYPE_CUSTOM.
            // This version string has the following format: OpenCL<space>C<space><major_version.minor_version><space><vendor-specific information>
            auto deviceSupportedOpenCLCVersion = reinterpret_cast<char*>(info);
            std::cout << str << " : " << deviceSupportedOpenCLCVersion << std::endl;
        }
            break;
        case CL_DEVICE_PROFILE: {
            // OpenCL profile string. Returns the profile name supported by the device (see note). The profile name returned can be one of the following strings:
            // FULL_PROFILE - if the device supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported).
            // EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile.
            // The platform profile returns the profile that is implemented by the OpenCL framework.
            // If the platform profile returned is FULL_PROFILE, the OpenCL framework will support devices that are FULL_PROFILE and may also support devices that are EMBEDDED_PROFILE.
            // The compiler must be available for all devices i.e. CL_DEVICE_COMPILER_AVAILABLE is CL_TRUE.
            // If the platform profile returned is EMBEDDED_PROFILE, then devices that are only EMBEDDED_PROFILE are supported.
            auto deviceSupportedOpenCLProfile = reinterpret_cast<char*>(info);
            std::cout << str << " : " << deviceSupportedOpenCLProfile << std::endl;
        }
            break;
        case CL_DEVICE_EXTENSIONS: {
            // Returns a space separated list of extension names (the extension names themselves do not contain any spaces) supported by the device.
            // The following approved Khronos extension names must be returned by all device that support OpenCL C 2.0:
            //      cl_khr_byte_addressable_store
            //      cl_khr_fp64 (for backward compatibility if double precision is supported)
            //      cl_khr_3d_image_writes
            //      cl_khr_image2d_from_buffer
            //      cl_khr_depth_images
            // Please refer to the OpenCL 2.0 Extension Specification for a detailed description of these extensions.
            auto deviceSupportedOpenCLExtensions = reinterpret_cast<char*>(info);
            std::cout << str << " : " << deviceSupportedOpenCLExtensions << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_PLATFORM: {
            // The platform associated with this device.
            auto devicePlatform = reinterpret_cast<cl_platform_id*>(info);

            size_t platformNameSize;
            checkOpenCLError(clGetPlatformInfo(*devicePlatform,CL_PLATFORM_NAME,0, nullptr,&platformNameSize));
            char* platformName = new char[platformNameSize];
            checkOpenCLError(clGetPlatformInfo(*devicePlatform,CL_PLATFORM_NAME,platformNameSize, platformName, nullptr));

            std::cout << str << " : " << platformName << std::endl;
        }
            break;
        case CL_DEVICE_AVAILABLE: {
            // Is CL_TRUE if the device is available and CL_FALSE otherwise.
            // A device is considered to be available if the device can be expected to successfully execute commands enqueued to the device.
            auto deviceAvailable = reinterpret_cast<cl_bool*>(info);
            std::cout << str << " : " << (deviceAvailable ? "Yes" : "No") << std::endl;
        }
            break;
        case CL_DEVICE_ADDRESS_BITS: {
            // The default compute device address space size of the global address space specified as an unsigned integer value in bits. Currently supported values are 32 or 64 bits.
            auto deviceAddressSpaceBits = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceAddressSpaceBits << "-bit device address space" << std::endl;
        }
            break;
        case CL_DEVICE_PROFILING_TIMER_RESOLUTION: {
            // The default compute device address space size of the global address space specified as an unsigned integer value in bits. Currently supported values are 32 or 64 bits.
            auto deviceTimerResolution = reinterpret_cast<size_t *>(info);
            std::cout << str << " : " << *deviceTimerResolution << " nanoseconds" << std::endl;
        }
        break;
        case CL_DEVICE_REFERENCE_COUNT: {
            // Returns the device reference count. If the device is a root-level device, a reference count of one is returned.
            auto deviceReferenceCount = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceReferenceCount << std::endl;
        }
            break;
        case CL_DEVICE_ENDIAN_LITTLE: {
            // Is CL_TRUE if the OpenCL device is a little endian device and CL_FALSE otherwise.
            auto deviceEndianLittle = reinterpret_cast<cl_bool *>(info);
            std::cout << str << " : " << (*deviceEndianLittle ? "Yes" : "No") << std::endl;
        }
            break;
        case CL_DEVICE_ERROR_CORRECTION_SUPPORT: {
            // Is CL_TRUE if the device implements error correction for all accesses to compute device memory (global and constant). Is CL_FALSE if the device does not implement such error correction.
            auto deviceErrorCorrSupport = reinterpret_cast<cl_bool *>(info);
            std::cout << str << " : " << (*deviceErrorCorrSupport ? "Yes" : "No") << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_EXECUTION_CAPABILITIES: {
            // Describes the execution capabilities of the device. This is a bit-field that describes one or more of the following values:
            // CL_EXEC_KERNEL - The OpenCL device can execute OpenCL kernels.
            // CL_EXEC_NATIVE_KERNEL - The OpenCL device can execute native kernels.
            // The mandated minimum capability is CL_EXEC_KERNEL
            auto deviceExecCapabilities = reinterpret_cast<cl_device_exec_capabilities*>(info);
            appendBitfield<cl_device_exec_capabilities>(*deviceExecCapabilities, CL_EXEC_KERNEL, "CL_EXEC_KERNEL", str);
            appendBitfield<cl_device_exec_capabilities>(*deviceExecCapabilities, CL_EXEC_NATIVE_KERNEL, "CL_EXEC_NATIVE_KERNEL", str);

            std::cout << str << std::endl;
        }
            break;
        case CL_DEVICE_COMPILER_AVAILABLE: {
            // Is CL_FALSE if the implementation does not have a compiler available to compile the program source.
            // Is CL_TRUE if the compiler is available. This can be CL_FALSE for the embedded platform profile only.
            auto deviceCompilerAvailable = reinterpret_cast<cl_bool*>(info);
            std::cout << str << " : " << (deviceCompilerAvailable ? "Yes" : "No") << std::endl;
        }
            break;
        case CL_DEVICE_LINKER_AVAILABLE: {
            // Is CL_FALSE if the implementation does not have a linker available. Is CL_TRUE if the linker is available.
            // This can be CL_FALSE for the embedded platform profile only. This must be CL_TRUE if CL_DEVICE_COMPILER_AVAILABLE is CL_TRUE.
            auto deviceLinkerAvailable = reinterpret_cast<cl_bool*>(info);
            std::cout << str << " : " << (deviceLinkerAvailable ? "Yes" : "No") << std::endl;
        }
            break;
        case CL_DEVICE_BUILT_IN_KERNELS: {
            // A semi-colon separated list of built-in kernels supported by the device. An empty string is returned if no built-in kernels are supported by the device.
            auto deviceBuiltInKernels = reinterpret_cast<char*>(info);
            std::cout << str << " : " << (strlen(deviceBuiltInKernels) == 0 ? "None" : deviceBuiltInKernels) << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_SINGLE_FP_CONFIG: {
            // Describes single precision floating-point capability of the device. This is a bit-field that describes one or more of the following values:
            // CL_FP_DENORM - denorms are supported
            // CL_FP_INF_NAN - INF and quiet NaNs are supported
            // CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported
            // CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported
            // CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported
            // CL_FP_FMA - IEEE754-2008 fused multiply-add is supported
            // CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT - divide and sqrt are correctly rounded as defined by the IEEE754 specification.
            // CL_FP_SOFT_FLOAT - Basic floating-point operations (such as addition, subtraction, multiplication) are implemented in software.
            // For the full profile, the mandated minimum floating-point capability for devices that are not of type CL_DEVICE_TYPE_CUSTOM is CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN.
            // For the embedded profile, see section 10.
            auto deviceSingleFpConfig = reinterpret_cast<cl_device_fp_config *>(info);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_DENORM, "CL_FP_DENORM", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_INF_NAN, "CL_FP_INF_NAN", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_ROUND_TO_NEAREST, "CL_FP_ROUND_TO_NEAREST", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_ROUND_TO_ZERO, "CL_FP_ROUND_TO_ZERO", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_ROUND_TO_INF, "CL_FP_ROUND_TO_INF", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_FMA, "CL_FP_FMA", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceSingleFpConfig, CL_FP_SOFT_FLOAT, "CL_FP_SOFT_FLOAT", str);

            std::cout << str << std::endl;
        }
            break;
        case CL_DEVICE_DOUBLE_FP_CONFIG: {
            // Describes double precision floating-point capability of the OpenCL device. This is a bit-field that describes one or more of the following values:
            //      CL_FP_DENORM - denorms are supported.
            //      CL_FP_INF_NAN - INF and NaNs are supported.
            //      CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
            //      CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
            //      CL_FP_ROUND_TO_INF - round to positive and negative infinity rounding modes supported.
            //      CL_FP_FMA - IEEE754-2008 fused multiply-add is supported.
            //      CL_FP_SOFT_FLOAT - Basic floating-point operations (such as addition, subtraction, multiplication) are implemented in software.
            //      Double precision is an optional feature so the mandated minimum double precision floating-point capability is 0.
            //      If double precision is supported by the device, then the minimum double precision floating-point capability must be: CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_DENORM.
            auto deviceDoubleFpConfig = reinterpret_cast<cl_device_fp_config *>(info);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_DENORM, "CL_FP_DENORM", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_INF_NAN, "CL_FP_INF_NAN", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_ROUND_TO_NEAREST, "CL_FP_ROUND_TO_NEAREST", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_ROUND_TO_ZERO, "CL_FP_ROUND_TO_ZERO", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_ROUND_TO_INF, "CL_FP_ROUND_TO_INF", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_FMA, "CL_FP_FMA", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceDoubleFpConfig, CL_FP_SOFT_FLOAT, "CL_FP_SOFT_FLOAT", str);

            std::cout << str << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_MAX_CLOCK_FREQUENCY: {
            // The default compute device address space size of the global address space specified as an unsigned integer value in bits. Currently supported values are 32 or 64 bits.
            auto deviceMaxClockFrequency = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceMaxClockFrequency << " Mhz" << std::endl;
        }
            break;
        case CL_DEVICE_MEM_BASE_ADDR_ALIGN: {
            // Alignment requirement (in bits) for sub-buffer offsets.
            // The minimum value is the size (in bits) of the largest OpenCL built-in data type supported by the device (long16 in FULL profile, long16 or int16 in EMBEDDED profile) for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
            auto deviceMemBaseAddrAlign = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceMemBaseAddrAlign << " bits" << std::endl;
        }
            break;
        case CL_DEVICE_MAX_COMPUTE_UNITS: {
            auto deviceNumCUs = reinterpret_cast<cl_uint*>(info);
            std::cout << str << " : " << *deviceNumCUs << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_GLOBAL_MEM_SIZE: {
            auto deviceGlobalMemSize = reinterpret_cast<cl_ulong*>(info);
            std::cout << str << " : " << *deviceGlobalMemSize << " bytes" << " | " << std::round(static_cast<double>(*deviceGlobalMemSize)/1024/1024) << " megabytes" << " | "
                      << std::round(static_cast<double>(*deviceGlobalMemSize)/1024/1024/1024) << " gigabytes" << std::endl;
        }
            break;
        case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: {
            auto deviceGlobalMemCacheSize = reinterpret_cast<cl_device_mem_cache_type*>(info);
            appendBitfield<cl_device_mem_cache_type>(*deviceGlobalMemCacheSize, CL_NONE, "CL_NONE", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceGlobalMemCacheSize, CL_READ_ONLY_CACHE, "CL_READ_ONLY_CACHE", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceGlobalMemCacheSize, CL_READ_WRITE_CACHE, "CL_READ_WRITE_CACHE", str);

            std::cout << str << std::endl;
        }
            break;
        case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: {
            auto deviceGlobalMemCacheSize = reinterpret_cast<cl_ulong*>(info);
            std::cout << str << " : " << *deviceGlobalMemCacheSize << " bytes" << " | " << static_cast<float>(*deviceGlobalMemCacheSize)/1024 << " kilobytes" <<
            " | " << static_cast<float>(*deviceGlobalMemCacheSize)/1024/1024 << " megabytes" << std::endl;
        }
            break;
        case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: {
            auto deviceGlobalMemCacheLineSize = reinterpret_cast<cl_uint*>(info);
            std::cout << str << " : " << *deviceGlobalMemCacheLineSize << " bytes" << std::endl;
        }
            break;
        case CL_DEVICE_LOCAL_MEM_TYPE: {
            // Type of local memory supported. This can be set to CL_LOCAL、CL_GLOBAL and CL_NONE.
            // CL_LOCAL: implying dedicated local memory storage such as SRAM.
            // CL_GLOBAL: implying there is no dedicated local memory, and global memory is treated as local memory.
            // CL_NONE: For custom devices, CL_NONE can also be returned indicating no local memory support.
            auto deviceLocalMemType = reinterpret_cast<cl_device_local_mem_type*>(info);
            appendBitfield<cl_device_mem_cache_type>(*deviceLocalMemType, CL_NONE, "CL_NONE", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceLocalMemType, CL_LOCAL, "CL_LOCAL", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceLocalMemType, CL_GLOBAL, "CL_GLOBAL", str);

            std::cout << str << std::endl;
        }
            break;
        case CL_DEVICE_LOCAL_MEM_SIZE: {
            // Size of local memory region in bytes. The minimum value is 32 KB for devices that are not of type CL_DEVICE_TYPE_CUSTOM
            auto deviceLocalMemSize = reinterpret_cast<cl_ulong*>(info);
            std::cout << str << " : " << *deviceLocalMemSize << " bytes" << " | " << static_cast<float>(*deviceLocalMemSize)/1024 << " kilobytes" << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_MAX_MEM_ALLOC_SIZE: {
            // Max size of memory object allocation in bytes.
            // The minimum value is max(min(1024*1024*1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE), 32*1024*1024) for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
            auto deviceMaxMemAllocSize = reinterpret_cast<cl_ulong*>(info);
            std::cout << str << " : " << *deviceMaxMemAllocSize << " bytes" << " | " << std::round(static_cast<double>(*deviceMaxMemAllocSize)/1024/1024) << " megabytes" << " | "
                      << std::round(static_cast<double>(*deviceMaxMemAllocSize)/1024/1024/1024) << " gigabytes" << std::endl;
        }
            break;
        case CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: {
            // The maximum number of bytes of storage that may be allocated for any single variable in program scope or inside a function in OpenCL C declared in the global address space.
            // The minimum value is 64 KB.
            auto deviceMaxGlobalVariableSize = reinterpret_cast<size_t*>(info);
            std::cout << str << " : " << *deviceMaxGlobalVariableSize << " bytes" << " | " << std::round(static_cast<double>(*deviceMaxGlobalVariableSize)/1024/1024) << " megabytes" << " | "
                      << std::round(static_cast<double>(*deviceMaxGlobalVariableSize)/1024/1024/1024) << " gigabytes" << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: {
            // Max size in bytes of a constant buffer allocation. The minimum value is 64 KB for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
            auto deviceMaxConstantBufferSize = reinterpret_cast<cl_ulong *>(info);
            std::cout << str << " : " << *deviceMaxConstantBufferSize << " bytes" << " | " << std::round(static_cast<double>(*deviceMaxConstantBufferSize)/1024) << " kilobytes" << std::endl;
        }
            break;
        case CL_DEVICE_MAX_CONSTANT_ARGS: {
            // Max number of arguments declared with the __constant qualifier in a kernel. The minimum value is 8 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
            auto deviceMaxConstantArgs = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceMaxConstantArgs << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_MAX_PARAMETER_SIZE: {
            // Max size in bytes of all arguments that can be passed to a kernel.
            // The minimum value is 1024 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
            // For this minimum value, only a maximum of 128 arguments can be passed to a kernel.
            auto deviceMaxParameterSize = reinterpret_cast<size_t *>(info);
            std::cout << str << " : " << *deviceMaxParameterSize << " bytes" << " | " << std::round(static_cast<double>(*deviceMaxParameterSize)/1024) << " kilobytes" << std::endl;
        }
            break;
        case CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: {
            // Maximum preferred total size, in bytes, of all program variables in the global address space. This is a performance hint.
            // An implementation may place such variables in storage with optimized device access. This query returns the capacity of such storage. The minimum value is 0.
            auto deviceGlobalVariablesPrefTotalSize = reinterpret_cast<size_t *>(info);
            std::cout << str << " : " << *deviceGlobalVariablesPrefTotalSize << " bytes" << " | " << std::round(static_cast<double>(*deviceGlobalVariablesPrefTotalSize)/1024) << " kilobytes" << std::endl;
        }
            break;
        case CL_DEVICE_PRINTF_BUFFER_SIZE: {
            // Maximum size in bytes of the internal buffer that holds the output of printf calls from a kernel. The minimum value for the FULL profile is 1 MB.
            auto deviceKernelPrintfBufferSize = reinterpret_cast<size_t *>(info);
            std::cout << str << " : " << *deviceKernelPrintfBufferSize << " bytes" << " | " << std::round(static_cast<double>(*deviceKernelPrintfBufferSize)/1024/1024) << " megabytes" << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_MAX_WORK_GROUP_SIZE: {
            // Maximum number of work-items in a work-group that a device is capable of executing on a single compute unit, for any given kernel-instance running on the device.
            // (Refer also to clEnqueueNDRangeKernel and CL_KERNEL_WORK_GROUP_SIZE). The minimum value is 1.
            auto deviceMaxWorkGroupSize = reinterpret_cast<size_t *>(info);
            std::cout << str << " : " << *deviceMaxWorkGroupSize << std::endl;
        }
            break;
        case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: {
            // Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. (Refer to clEnqueueNDRangeKernel).
            // The minimum value is 3 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
            auto deviceMaxWorkItemDim = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceMaxWorkItemDim << std::endl;
        }
            break;
        case CL_DEVICE_MAX_WORK_ITEM_SIZES: {
            // Maximum number of work-items that can be specified in each dimension of the work-group to clEnqueueNDRangeKernel.
            // Returns n size_t entries, where n is the value returned by the query for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS.
            // The minimum value is (1, 1, 1) for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
            auto deviceMaxWorkItemSizes = reinterpret_cast<size_t *>(info);

            cl_uint deviceMaxWorkItemDim;
            checkOpenCLError( clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(deviceMaxWorkItemDim), &deviceMaxWorkItemDim, nullptr) );

            std::cout << str << " : ";
            for(cl_uint idx = 0; idx < deviceMaxWorkItemDim; ++idx) {
                if (idx == 0) {
                    std::cout << "( " << deviceMaxWorkItemSizes[idx] << ", ";
                }
                else if (idx == deviceMaxWorkItemDim - 1) {
                    std::cout << deviceMaxWorkItemSizes[idx] << " )" << std::endl;
                }
                else {
                    std::cout << deviceMaxWorkItemSizes[idx] << ", ";
                }
            }
        }
            break;
        case CL_DEVICE_MAX_NUM_SUB_GROUPS: {
            // Maximum number of sub-groups in a work-group that a device is capable of executing on a single compute unit, for any given kernel-instance running on the device.
            // The minimum value is 1. (Refer also to clGetKernelSubGroupInfo.)
            auto deviceMaxNumSubGroups = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceMaxNumSubGroups << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_MAX_ON_DEVICE_QUEUES: {
            // The maximum number of device queues that can be created per context. The minimum value is 1.
            auto deviceCommandQueueMaxCount = reinterpret_cast<cl_uint *>(info);

            std::cout << str << " : " << *deviceCommandQueueMaxCount << std::endl;
        }
            break;
        case CL_DEVICE_MAX_ON_DEVICE_EVENTS: {
            // The maximum number of events in use by a device queue.
            // These refer to events returned by the enqueue_ built-in functions to a device queue or user events returned by the create_user_event built-in function that have not been released.
            // The minimum value is 1024.
            auto deviceCommandQueueMaxEvents = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceCommandQueueMaxEvents << std::endl;
        }
            break;
        case CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: {
            // The max size of the device queue in bytes. The minimum value is 256 KB for the full profile and 64 KB for the embedded profile
            auto deviceCommandQueueOnDeviceMaxSize = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceCommandQueueOnDeviceMaxSize << " bytes" << " | " << static_cast<float>(*deviceCommandQueueOnDeviceMaxSize)/1024 << " kilobytes" << std::endl;
        }
            break;
        case CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: {
            // The size of the device queue in bytes preferred by the implementation. Applications should use this size for the device queue to ensure good performance. The minimum value is 16 KB.
            auto deviceCommandQueueOnDevicePrefSize = reinterpret_cast<cl_uint *>(info);
            std::cout << str << " : " << *deviceCommandQueueOnDevicePrefSize << " bytes" << " | " << static_cast<float>(*deviceCommandQueueOnDevicePrefSize)/1024 << " kilobytes" << std::endl;
        }
            break;
        case CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: {
            // Describes the on device command-queue properties supported by the device. This is a bit-field that describes one or more of the following values:
            //      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
            //      CL_QUEUE_PROFILING_ENABLE
            // These properties are described in the table for clCreateCommandQueueWithProperties. The mandated minimum capability is CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE..
            auto deviceCommandQueueOnDeviceProperties = reinterpret_cast<cl_command_queue_properties *>(info);
            appendBitfield<cl_device_mem_cache_type>(*deviceCommandQueueOnDeviceProperties, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceCommandQueueOnDeviceProperties, CL_QUEUE_PROFILING_ENABLE, "CL_QUEUE_PROFILING_ENABLE", str);

            std::cout << str << std::endl;
        }
            break;
        case CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: {
            // Describes the on host command-queue properties supported by the device. This is a bit-field that describes one or more of the following values:
            //      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
            //      CL_QUEUE_PROFILING_ENABLE
            // These properties are described in the table for clCreateCommandQueueWithProperties. The mandated minimum capability is CL_QUEUE_PROFILING_ENABLE.
            auto deviceCommandQueueOnHostProperties = reinterpret_cast<cl_command_queue_properties *>(info);
            appendBitfield<cl_device_mem_cache_type>(*deviceCommandQueueOnHostProperties, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", str);
            appendBitfield<cl_device_mem_cache_type>(*deviceCommandQueueOnHostProperties, CL_QUEUE_PROFILING_ENABLE, "CL_QUEUE_PROFILING_ENABLE", str);

            std::cout << str << std::endl;
        }
            break;
        // ...
        case CL_DEVICE_IMAGE_SUPPORT: {
            // Is CL_TRUE if images are supported by the OpenCL device and CL_FALSE otherwise.
            auto deviceImageSupport = reinterpret_cast<cl_bool *>(info);

            std::cout << str << " : " << (*deviceImageSupport ? "Yes" : "No") << std::endl;
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
