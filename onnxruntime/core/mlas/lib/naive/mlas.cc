#pragma once

#include <mlas.h>
#include <algorithm>

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
    ){}

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

if (TransA == CblasNoTrans) {

            if (TransB == CblasNoTrans) {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const T* a = A + (m * lda);
                        const T* b = B + n;
                        T* c = C + (m * ldc) + n;
                        T sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += ldb;
                            a += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
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

                        *c = (*c * beta) + (sum * alpha);
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

                        *c = (*c * beta) + (sum * alpha);
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

                        *c = (*c * beta) + (sum * alpha);
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

}

//
// Convolution routines.
//

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
    ){}

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

        int64_t KernelHeight = KernelShape[0];
        int64_t KernelWidth = KernelShape[1];

        int64_t PaddingLeftY = Padding[0];
        int64_t PaddingLeftX = Padding[1];
        int64_t PaddingRightY = Padding[2];
        int64_t PaddingRightX = Padding[3];

        int64_t StrideHeight = StrideShape[0];
        int64_t StrideWidth = StrideShape[1];

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

        int64_t KernelHeight = KernelShape[0];
        int64_t KernelWidth = KernelShape[1];

        int64_t PaddingLeftY = Padding[0];
        int64_t PaddingLeftX = Padding[1];
        int64_t PaddingRightY = Padding[2];
        int64_t PaddingRightX = Padding[3];

        int64_t StrideHeight = StrideShape[0];
        int64_t StrideWidth = StrideShape[1];

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
    if (PoolingKind == MlasMaximumPooling) {
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
    ){}

void
MLASCALL
MlasComputeTanh(
    const float* Input,
    float* Output,
    size_t N
    ){}

void
MLASCALL
MlasComputeErf(
    const float* Input,
    float* Output,
    size_t N
    ){}

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
    ){}

//
// Buffer reordering routines.
//

void
MLASCALL
MlasReorderInput(
    const int64_t* InputShape,
    const float* S,
    float* D
    ){}

void
MLASCALL
MlasReorderOutputNchw(
    const int64_t* OutputShape,
    const float* S,
    float* D
    ){}

void
MLASCALL
MlasReorderOutputNhwc(
    const int64_t* OutputShape,
    const float* S,
    float* D
    ){}

void
MLASCALL
MlasReorderFilterOIHWBiBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    ){}

void
MLASCALL
MlasReorderFilterOIHWBo(
    const int64_t* FilterShape,
    const float* S,
    float* D
    ){}

//
// Single precision NCHWc routines.
//

size_t
MLASCALL
MlasNchwcGetBlockSize(
    void
    ){
    return 1;
}

void
MLASCALL
MlasNchwcConv(
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
    ){}

void
MLASCALL
MlasNchwcPool(
    MLAS_POOLING_KIND PoolingKind,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    ){}

void
MLASCALL
MlasNchwcUpsample(
    const int64_t* InputShape,
    const int64_t* Scales,
    const float* Input,
    float* Output
    ){}

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
    ){}

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
    ){}

template
void
MLASCALL
MlasQuantizeLinear<int8_t>(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    );

template
void
MLASCALL
MlasQuantizeLinear<uint8_t>(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );

