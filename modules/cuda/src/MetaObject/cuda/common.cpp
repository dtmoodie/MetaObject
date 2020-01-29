#include "common.hpp"

namespace mo
{
}

#define PRINT_CUDA_ERR(ERR)                                                                                            \
    case ERR:                                                                                                          \
        return #ERR

namespace fmt
{
    std::string to_string(const cudaError err)
    {
        switch (err)
        {
            PRINT_CUDA_ERR(cudaSuccess);
            PRINT_CUDA_ERR(cudaErrorMissingConfiguration);
            PRINT_CUDA_ERR(cudaErrorMemoryAllocation);
            PRINT_CUDA_ERR(cudaErrorInitializationError);
            PRINT_CUDA_ERR(cudaErrorLaunchFailure);
            PRINT_CUDA_ERR(cudaErrorPriorLaunchFailure);
            PRINT_CUDA_ERR(cudaErrorLaunchTimeout);
            PRINT_CUDA_ERR(cudaErrorLaunchOutOfResources);
            PRINT_CUDA_ERR(cudaErrorInvalidDeviceFunction);
            PRINT_CUDA_ERR(cudaErrorInvalidConfiguration);
            PRINT_CUDA_ERR(cudaErrorInvalidDevice);
            PRINT_CUDA_ERR(cudaErrorInvalidValue);
            PRINT_CUDA_ERR(cudaErrorInvalidPitchValue);
            PRINT_CUDA_ERR(cudaErrorInvalidSymbol);
            PRINT_CUDA_ERR(cudaErrorMapBufferObjectFailed);
            PRINT_CUDA_ERR(cudaErrorUnmapBufferObjectFailed);
            PRINT_CUDA_ERR(cudaErrorInvalidHostPointer);
            PRINT_CUDA_ERR(cudaErrorInvalidDevicePointer);
            PRINT_CUDA_ERR(cudaErrorInvalidTexture);
            PRINT_CUDA_ERR(cudaErrorInvalidTextureBinding);
            PRINT_CUDA_ERR(cudaErrorInvalidChannelDescriptor);
            PRINT_CUDA_ERR(cudaErrorInvalidMemcpyDirection);
            PRINT_CUDA_ERR(cudaErrorAddressOfConstant);
            PRINT_CUDA_ERR(cudaErrorTextureFetchFailed);
            PRINT_CUDA_ERR(cudaErrorTextureNotBound);
            PRINT_CUDA_ERR(cudaErrorSynchronizationError);
            PRINT_CUDA_ERR(cudaErrorInvalidFilterSetting);
            PRINT_CUDA_ERR(cudaErrorInvalidNormSetting);
            PRINT_CUDA_ERR(cudaErrorMixedDeviceExecution);
            PRINT_CUDA_ERR(cudaErrorCudartUnloading);
            PRINT_CUDA_ERR(cudaErrorUnknown);
            PRINT_CUDA_ERR(cudaErrorNotYetImplemented);
            PRINT_CUDA_ERR(cudaErrorMemoryValueTooLarge);
            PRINT_CUDA_ERR(cudaErrorInvalidResourceHandle);
            PRINT_CUDA_ERR(cudaErrorNotReady);
            PRINT_CUDA_ERR(cudaErrorInsufficientDriver);
            PRINT_CUDA_ERR(cudaErrorSetOnActiveProcess);
            PRINT_CUDA_ERR(cudaErrorInvalidSurface);
            PRINT_CUDA_ERR(cudaErrorNoDevice);
            PRINT_CUDA_ERR(cudaErrorECCUncorrectable);
            PRINT_CUDA_ERR(cudaErrorSharedObjectSymbolNotFound);
            PRINT_CUDA_ERR(cudaErrorSharedObjectInitFailed);
            PRINT_CUDA_ERR(cudaErrorUnsupportedLimit);
            PRINT_CUDA_ERR(cudaErrorDuplicateVariableName);
            PRINT_CUDA_ERR(cudaErrorDuplicateTextureName);
            PRINT_CUDA_ERR(cudaErrorDuplicateSurfaceName);
            PRINT_CUDA_ERR(cudaErrorDevicesUnavailable);
            PRINT_CUDA_ERR(cudaErrorInvalidKernelImage);
            PRINT_CUDA_ERR(cudaErrorNoKernelImageForDevice);
            PRINT_CUDA_ERR(cudaErrorIncompatibleDriverContext);
            PRINT_CUDA_ERR(cudaErrorPeerAccessAlreadyEnabled);
            PRINT_CUDA_ERR(cudaErrorPeerAccessNotEnabled);
            PRINT_CUDA_ERR(cudaErrorDeviceAlreadyInUse);
            PRINT_CUDA_ERR(cudaErrorProfilerDisabled);

            PRINT_CUDA_ERR(cudaErrorProfilerNotInitialized);
            PRINT_CUDA_ERR(cudaErrorProfilerAlreadyStarted);
            PRINT_CUDA_ERR(cudaErrorProfilerAlreadyStopped);
            PRINT_CUDA_ERR(cudaErrorAssert);
            PRINT_CUDA_ERR(cudaErrorTooManyPeers);
            PRINT_CUDA_ERR(cudaErrorHostMemoryAlreadyRegistered);
            PRINT_CUDA_ERR(cudaErrorHostMemoryNotRegistered);
            PRINT_CUDA_ERR(cudaErrorOperatingSystem);
            PRINT_CUDA_ERR(cudaErrorPeerAccessUnsupported);
            PRINT_CUDA_ERR(cudaErrorLaunchMaxDepthExceeded);
            PRINT_CUDA_ERR(cudaErrorLaunchFileScopedTex);
            PRINT_CUDA_ERR(cudaErrorLaunchFileScopedSurf);

            PRINT_CUDA_ERR(cudaErrorSyncDepthExceeded);
            PRINT_CUDA_ERR(cudaErrorLaunchPendingCountExceeded);
            PRINT_CUDA_ERR(cudaErrorNotPermitted);
            PRINT_CUDA_ERR(cudaErrorNotSupported);
            PRINT_CUDA_ERR(cudaErrorHardwareStackError);
            PRINT_CUDA_ERR(cudaErrorIllegalInstruction);
            PRINT_CUDA_ERR(cudaErrorMisalignedAddress);
            PRINT_CUDA_ERR(cudaErrorInvalidAddressSpace);
            PRINT_CUDA_ERR(cudaErrorInvalidPc);
            PRINT_CUDA_ERR(cudaErrorIllegalAddress);
            PRINT_CUDA_ERR(cudaErrorInvalidPtx);
            PRINT_CUDA_ERR(cudaErrorInvalidGraphicsContext);
#if CUDART_VERSION > 8000
            PRINT_CUDA_ERR(cudaErrorNvlinkUncorrectable);
            PRINT_CUDA_ERR(cudaErrorJitCompilerNotFound);
            PRINT_CUDA_ERR(cudaErrorCooperativeLaunchTooLarge);
#endif
            PRINT_CUDA_ERR(cudaErrorStartupFailure);
            PRINT_CUDA_ERR(cudaErrorApiFailureBase);
#if CUDART_VERSION >= 10000
            PRINT_CUDA_ERR(cudaErrorSystemNotReady);
            PRINT_CUDA_ERR(cudaErrorStreamCaptureImplicit);
            PRINT_CUDA_ERR(cudaErrorIllegalState);
            PRINT_CUDA_ERR(cudaErrorStreamCaptureUnsupported);
            PRINT_CUDA_ERR(cudaErrorStreamCaptureInvalidated);
            PRINT_CUDA_ERR(cudaErrorStreamCaptureMerge);
            PRINT_CUDA_ERR(cudaErrorStreamCaptureUnmatched);
            PRINT_CUDA_ERR(cudaErrorStreamCaptureUnjoined);
            PRINT_CUDA_ERR(cudaErrorStreamCaptureIsolation);
            PRINT_CUDA_ERR(cudaErrorCapturedEvent);
#endif
        }
        return "No conversion for this error code";
    }
}

namespace std
{
    ostream& operator<<(ostream& os, cudaError err)
    {
        os << fmt::to_string(err);
        return os;
    }
}
