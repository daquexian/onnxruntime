#pragma once

#include <mlas.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#define UNIMPLEMENTED throw std::runtime_error(__FILE__ + std::string(" ") + std::to_string(__LINE__) + std::string(" unimplemented"))

MLAS_THREADPOOL* threadpool = nullptr;

template <typename T>
class MatrixGuardBuffer
{
public:
    MatrixGuardBuffer()
    {
        _BaseBuffer = nullptr;
        _BaseBufferSize = 0;
        _ElementsAllocated = 0;
    }

    ~MatrixGuardBuffer(void)
    {
        ReleaseBuffer();
    }

    T* GetBuffer(size_t Elements)
    {
        //
        // Check if the internal buffer needs to be reallocated.
        //

        if (Elements > _ElementsAllocated) {

            std::cout << __LINE__ << std::endl;
            ReleaseBuffer();

            std::cout << __LINE__ << std::endl;
            //
            // Reserve a virtual address range for the allocation plus an unmapped
            // guard region.
            //

            constexpr size_t BufferAlignment = 64 * 1024;
            constexpr size_t GuardPadding = 256 * 1024;

            size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

            _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
            _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
            std::cout << __LINE__ << std::endl;
            // _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            _BaseBuffer = malloc(_BaseBufferSize);
            std::cout << _BaseBufferSize << std::endl;
            std::cout << __LINE__ << std::endl;
#endif

            std::cout << __LINE__ << std::endl;
            if (_BaseBuffer == nullptr) {
            std::cout << "bad!" << std::endl;
                throw std::bad_alloc();
            }
            std::cout << __LINE__ << std::endl;

            //
            // Commit the number of bytes for the allocation leaving the upper
            // guard region as unmapped.
            //

#if defined(_WIN32)
            if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
                throw std::bad_alloc();
            }
#else
            std::cout << __LINE__ << std::endl;
            // if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
                // throw std::bad_alloc();
            // }
            std::cout << __LINE__ << std::endl;
#endif

            std::cout << __LINE__ << std::endl;
            _ElementsAllocated = BytesToAllocate / sizeof(T);
            _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
            std::cout << __LINE__ << std::endl;
        }

        //
        //
        //

        T* GuardAddress = _GuardAddress;
        T* buffer = GuardAddress - Elements;

        const int MinimumFillValue = -23;
        const int MaximumFillValue = 23;

        int FillValue = MinimumFillValue;
        T* FillAddress = buffer;

            std::cout << __LINE__ << std::endl;
        while (FillAddress < GuardAddress) {

            *FillAddress++ = (T)FillValue;

            FillValue++;

            if (FillValue > MaximumFillValue) {
                FillValue = MinimumFillValue;
            }
        }

            std::cout << __LINE__ << std::endl;
        return buffer;
    }

    void ReleaseBuffer(void)
    {
        if (_BaseBuffer != nullptr) {

#if defined(_WIN32)
            VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
            std::cout << __LINE__ << std::endl;
            // munmap(_BaseBuffer, _BaseBufferSize);
            free(_BaseBuffer);
            std::cout << __LINE__ << std::endl;
#endif

            _BaseBuffer = nullptr;
            _BaseBufferSize = 0;
        }

        _ElementsAllocated = 0;
    }

private:
    size_t _ElementsAllocated;
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    T* _GuardAddress;
};

size_t
MLASCALL
MlasGetPreferredBufferAlignment(
    void
    )
/*++

Routine Description:

    This routine returns the preferred byte alignment for buffers that are used
    with this library. Buffers that are not byte aligned to this value will
    function, but will not achieve best performance.

Arguments:

    None.

Return Value:

    Returns the preferred byte alignment for buffers.

--*/
{
    return 64;
}

void
MLASCALL
MlasActivation(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc
    ){

    switch (Activation->ActivationKind) {
        case MlasReluActivation:
            {
            int x = 0;
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    if (Bias != nullptr) {
                        Buffer[x+j] += Bias[i];
                    }
                    Buffer[x+j] = std::max(Buffer[x+j], 0.F);
                }
                x += ldc;
            }
            break;
            }
        case MlasIdentityActivation:
            {
            int x = 0;
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    if (Bias != nullptr) {
                        Buffer[x+j] += Bias[i];
                        if (std::isnan(Buffer[x+j])) {
                            std::cout << ("Buffer[x+j] nan " + std::to_string(i) + " " + std::to_string(j)) << std::endl;
                            throw std::runtime_error("Buffer[x+j] nan " + std::to_string(i) + " " + std::to_string(j));
                        }
                    }
                }
                x += ldc;
            }
            break;
            }
        default:
            UNIMPLEMENTED;
    }
}

//
// Matrix/matrix multiply routines.
//

template<class T>
void ReferenceGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const T* A,
    size_t lda,
    const T* B,
    size_t ldb,
    float beta,
    T* C,
    size_t ldc
    ) {

    // if (beta != 0 || alpha != 1) {
    //     UNIMPLEMENTED;
    // }

if (TransA == CblasNoTrans) {

            if (TransB == CblasNoTrans) {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + (m * lda);
                        const T* b = B + n;
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            // if (m == 0 || n == 98) {
                            //     std::cout << "b: " << *b << ", a: " << *a << std::endl;
                            //     std::cout << "sum before adding: " << sum << std::endl;
                            // }
                            sum += (*b * *a);
                            // if (m == 0 || n == 98) {
                            //     std::cout << "sum after adding: " << sum << std::endl;
                            // }
                            b += ldb;
                            a += 1;
                        }

                            // if (m == 0 || n == 98) {
                            //     std::cout << "sum: " << sum << std::endl;
                            // }

                        // *c = (*c * beta) + (sum * alpha);

                        if (beta != 0) {
                            *c *= beta;
                            if (alpha != 1) {
                                *c += sum * alpha;
                            } else {
                                *c += sum;
                            }
                        } else {
                            if (alpha != 1) {
                                *c = sum * alpha;
                            } else {
                                *c = sum;
                            }
                        }

                        if (std::isnan(*c)) {
                            std::cout << ("c nan " + std::to_string(m) + " " + std::to_string(n)) << std::endl;
                            throw std::runtime_error("c nan " + std::to_string(m) + " " + std::to_string(n));
                        }

                    }
                }

            } else {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + (m * lda);
                        const T* b = B + (n * ldb);
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += 1;
                            a += 1;
                        }

                        // *c = (*c * beta) + (sum * alpha);
                        if (beta != 0) {
                            *c *= beta;
                            if (alpha != 1) {
                                *c += sum * alpha;
                            } else {
                                *c += sum;
                            }
                        } else {
                            if (alpha != 1) {
                                *c = sum * alpha;
                            } else {
                                *c = sum;
                            }
                        }

                        if (std::isnan(*c)) {
                            std::cout << (__LINE__ + std::string("c nan ") + std::to_string(m) + " " + std::to_string(n)) << std::endl;
                            throw std::runtime_error("c nan " + std::to_string(m) + " " + std::to_string(n));
                        }
                    }
                }
            }

        } else {

            if (TransB == CblasNoTrans) {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + m;
                        const T* b = B + n;
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += ldb;
                            a += lda;
                        }

                        // *c = (*c * beta) + (sum * alpha);
                        if (beta != 0) {
                            *c *= beta;
                            if (alpha != 1) {
                                *c += sum * alpha;
                            } else {
                                *c += sum;
                            }
                        } else {
                            if (alpha != 1) {
                                *c = sum * alpha;
                            } else {
                                *c = sum;
                            }
                        }

                        if (std::isnan(*c)) {
                            std::cout << (__LINE__ + std::string("c nan ") + std::to_string(m) + " " + std::to_string(n)) << std::endl;
                            throw std::runtime_error("c nan " + std::to_string(m) + " " + std::to_string(n));
                        }
                    }
                }

            } else {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + m;
                        const T* b = B + (n * ldb);
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += 1;
                            a += lda;
                        }

                        // *c = (*c * beta) + (sum * alpha);
                        if (beta != 0) {
                            *c *= beta;
                            if (alpha != 1) {
                                *c += sum * alpha;
                            } else {
                                *c += sum;
                            }
                        } else {
                            if (alpha != 1) {
                                *c = sum * alpha;
                            } else {
                                *c = sum;
                            }
                        }

                        if (std::isnan(*c)) {
                            std::cout << (__LINE__ + std::string("c nan ") + std::to_string(m) + " " + std::to_string(n)) << std::endl;
                            throw std::runtime_error("c nan " + std::to_string(m) + " " + std::to_string(n));
                        }
                    }
                }
            }
        }
}

void
MLASCALL
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    ){
    ReferenceGemm<float>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
MLASCALL
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    double alpha,
    const double* A,
    size_t lda,
    const double* B,
    size_t ldb,
    double beta,
    double* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    ){
    ReferenceGemm<double>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const int8_t* B,
    size_t ldb,
    int8_t offb,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    ){

            UNIMPLEMENTED;
}

void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const uint8_t* B,
    size_t ldb,
    uint8_t offb,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    ){

            UNIMPLEMENTED;
}

//
// Convolution routines.
//

    MatrixGuardBuffer<float> BufferIm2Col;
    void
    ReferenceConv2D(
        size_t BatchCount,
        size_t GroupCount,
        size_t InputChannels,
        size_t InputHeight,
        size_t InputWidth,
        size_t FilterCount,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t DilationHeight,
        size_t DilationWidth,
        size_t StrideHeight,
        size_t StrideWidth,
        size_t OutputHeight,
        size_t OutputWidth,
        const float* Input,
        const float* Filter,
        const float* Bias,
        float* Output
        )
    {
        size_t InputSize = InputHeight * InputWidth;
        size_t OutputSize = OutputHeight * OutputWidth;
        size_t KernelSize = KernelHeight * KernelWidth;

        size_t K = InputChannels * KernelSize;
        size_t Im2ColElements = OutputSize * K;

        for (size_t b = 0; b < BatchCount; b++) {

            const float* filter = Filter;
            const float* bias = Bias;

            for (size_t g = 0; g < GroupCount; g++) {

                //
                // Transform the image using IM2COL and invoke the GEMM.
                //

                float* Im2Col = BufferIm2Col.GetBuffer(Im2ColElements);
                float* Im2ColOut = Im2Col;

                for (size_t c = 0; c < InputChannels; c++) {

                    for (size_t ky = 0; ky < KernelHeight; ky++) {

                        for (size_t kx = 0; kx < KernelWidth; kx++) {

                            for (size_t oh = 0; oh < OutputHeight; oh++) {

                                size_t ih = oh * StrideHeight + ky * DilationHeight - PaddingLeftHeight;

                                for (size_t ow = 0; ow < OutputWidth; ow++) {

                                    size_t iw = ow * StrideWidth + kx * DilationWidth - PaddingLeftWidth;

                                    *Im2ColOut++ = (ih < InputHeight && iw < InputWidth) ?
                                        Input[ih * InputWidth + iw] : 0;
                                }
                            }
                        }
                    }

                    Input += InputSize;
                }

                MlasGemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f,
                    filter, K, Im2Col, OutputSize, 0.0f, Output, OutputSize, threadpool);

                //
                // Apply the bias.
                //

                for (size_t f = 0; f < FilterCount; f++) {

                    float biasValue = *bias++;

                    for (size_t o = 0; o < OutputSize; o++) {
                        *Output++ += biasValue;
                    }
                }

                filter += FilterCount * InputChannels * KernelSize;
            }
        }
    }
void
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    size_t BatchCount,
    size_t GroupCount,
    size_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t FilterCount,
    const MLAS_ACTIVATION* Activation,
    size_t* WorkingBufferSize,
    MLAS_THREADPOOL* ThreadPool
    ){
    *WorkingBufferSize = 0;
}

void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    ){
            UNIMPLEMENTED;
}

//
// Pooling routines.
//

    void
    ReferenceMaximumPool2D(
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const float* Input,
        float* Output
        )
    {
        int64_t ChannelCount = InputShape[0] * InputShape[1];

        int64_t InputHeight = InputShape[2];
        int64_t InputWidth = InputShape[3];

        int64_t KernelHeight = KernelShape ? KernelShape[0] : InputHeight;
        int64_t KernelWidth = KernelShape ? KernelShape[1] : InputWidth;

        int64_t PaddingLeftY = Padding ? Padding[0] : 0;
        int64_t PaddingLeftX = Padding ? Padding[1] : 0;
        int64_t PaddingRightY = Padding ? Padding[2] : 0;
        int64_t PaddingRightX = Padding ? Padding[3] : 0;

        int64_t StrideHeight = StrideShape ? StrideShape[0] : 1;
        int64_t StrideWidth = StrideShape ? StrideShape[1] : 1;

        int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
        int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

        for (int64_t c = 0; c < ChannelCount; c++) {

            for (int64_t ph = 0; ph < OutputHeight; ph++) {

                int64_t ihStart = ph * StrideHeight - PaddingLeftY;
                int64_t ihEnd = ihStart + KernelHeight;

                ihStart = (std::max)(ihStart, int64_t(0));
                ihEnd = (std::min)(ihEnd, InputHeight);

                for (int64_t pw = 0; pw < OutputWidth; pw++) {

                    int64_t iwStart = pw * StrideWidth - PaddingLeftX;
                    int64_t iwEnd = iwStart + KernelWidth;

                    iwStart = (std::max)(iwStart, int64_t(0));
                    iwEnd = (std::min)(iwEnd, InputWidth);

                    float m = std::numeric_limits<float>::lowest();

                    for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                        for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                            m = (std::max)(m, Input[ih * InputWidth + iw]);
                        }
                    }

                    Output[ph * OutputWidth + pw] = m;
                }
            }

            Input += InputHeight * InputWidth;
            Output += OutputHeight * OutputWidth;
        }
    }

    void
    ReferenceAveragePool2D(
        const int64_t* InputShape,
        const int64_t* KernelShape,
        const int64_t* Padding,
        const int64_t* StrideShape,
        const float* Input,
        float* Output,
        bool CountIncludePad
        )
    {
        int64_t ChannelCount = InputShape[0] * InputShape[1];

        int64_t InputHeight = InputShape[2];
        int64_t InputWidth = InputShape[3];

        int64_t KernelHeight = KernelShape ? KernelShape[0] : InputHeight;
        int64_t KernelWidth = KernelShape ? KernelShape[1] : InputWidth;

        int64_t PaddingLeftY = Padding ? Padding[0] : 0;
        int64_t PaddingLeftX = Padding ? Padding[1] : 0;
        int64_t PaddingRightY = Padding ? Padding[2] : 0;
        int64_t PaddingRightX = Padding ? Padding[3] : 0;

        int64_t StrideHeight = StrideShape ? StrideShape[0] : 1;
        int64_t StrideWidth = StrideShape ? StrideShape[1] : 1;

        int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
        int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

        for (int64_t c = 0; c < ChannelCount; c++) {

            for (int64_t ph = 0; ph < OutputHeight; ph++) {

                int64_t ihStart = ph * StrideHeight - PaddingLeftY;
                int64_t ihEnd = ihStart + KernelHeight;

                ihStart = (std::max)(ihStart, int64_t(0));
                ihEnd = (std::min)(ihEnd, InputHeight);

                for (int64_t pw = 0; pw < OutputWidth; pw++) {

                    int64_t iwStart = pw * StrideWidth - PaddingLeftX;
                    int64_t iwEnd = iwStart + KernelWidth;

                    iwStart = (std::max)(iwStart, int64_t(0));
                    iwEnd = (std::min)(iwEnd, InputWidth);

                    float m = 0.0f;

                    for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                        for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                            m += Input[ih * InputWidth + iw];
                        }
                    }

                    if (CountIncludePad) {
                        m /= (KernelHeight * KernelWidth);
                    } else {
                        m /= (ihEnd - ihStart) * (iwEnd - iwStart);
                    }

                    Output[ph * OutputWidth + pw] = m;
                }
            }

            Input += InputHeight * InputWidth;
            Output += OutputHeight * OutputWidth;
        }
    }
void
MLASCALL
MlasPool(
    MLAS_POOLING_KIND PoolingKind,
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    ){
    if (Dimensions != 2) {
        throw std::runtime_error("unimplemented");
    }
    if (PoolingKind == MlasMaximumPooling) {
        ReferenceMaximumPool2D(InputShape, KernelShape, Padding, StrideShape, Input, Output);
    } else if (PoolingKind == MlasAveragePoolingExcludePad) {
        ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, Output, false);
    } else if (PoolingKind == MlasAveragePoolingIncludePad) {
        ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, Output, true);
    } else {
        throw std::runtime_error("unimplemented");
    }
}

//
// Miscellaneous compute routines.
//

void
MLASCALL
MlasComputeLogistic(
    const float* Input,
    float* Output,
    size_t N
    ){
    for (size_t i = 0; i < N; i++){
        Output[i] = 1 / (1 + std::exp(-Input[i]));
    }
}

void
MLASCALL
MlasComputeTanh(
    const float* Input,
    float* Output,
    size_t N
    ){
    for (size_t i = 0; i < N; i++){
        Output[i] = std::tanh(Input[i]);
    }
}

void
MLASCALL
MlasComputeErf(
    const float* Input,
    float* Output,
    size_t N
    ){
    for (size_t i = 0; i < N; i++){
        Output[i] = std::erf(Input[i]);
    }
}

//
// Half-precision floating-point routines.
//

extern "C"
void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const unsigned short* Source,
    float* Destination,
    size_t Count
    ){

            UNIMPLEMENTED;
}

//
// Buffer reordering routines.
//

void
MLASCALL
MlasReorderInput(
    const int64_t* InputShape,
    const float* S,
    float* D
    ){

            UNIMPLEMENTED;
}

void
MLASCALL
MlasReorderOutputNchw(
    const int64_t* OutputShape,
    const float* S,
    float* D
    ){

            UNIMPLEMENTED;
}

void
MLASCALL
MlasReorderOutputNhwc(
    const int64_t* OutputShape,
    const float* S,
    float* D
    ){

            UNIMPLEMENTED;
}

void
MLASCALL
MlasReorderFilterOIHWBiBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    ){

            UNIMPLEMENTED;
}

void
MLASCALL
MlasReorderFilterOIHWBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    ){

            UNIMPLEMENTED;
}

//
// Single precision NCHWc routines.
//

size_t
MLASCALL
MlasNchwcGetBlockSize(
    void
    ){
            UNIMPLEMENTED;
}

void
MLASCALL
MlasNchwcConv(
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t GroupCount,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    bool ZeroMode,
    MLAS_THREADPOOL* ThreadPool
    ) {

            UNIMPLEMENTED;
}

void
MLASCALL
MlasNchwcPool(
    MLAS_POOLING_KIND PoolingKind,
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    ) {

            UNIMPLEMENTED;
}

void
MLASCALL
MlasNchwcUpsample(
    const int64_t* InputShape,
    const int64_t* Scales,
    const float* Input,
    float* Output
    ){

            UNIMPLEMENTED;
}

//
// Linear quantization routines.
//

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    ){

            UNIMPLEMENTED;
}

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    ){

            UNIMPLEMENTED;
}

void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    ) {

            UNIMPLEMENTED;
}

void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    ) {

            UNIMPLEMENTED;
}
