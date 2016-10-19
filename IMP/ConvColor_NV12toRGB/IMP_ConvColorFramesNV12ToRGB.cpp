#include "IMP_ConvColorFramesNV12ToRGB.hpp"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"
#include "opencv2/cudaimgproc.hpp"

#if defined (CLIENT)
#include "..\VPL\QT_simple\VPL_FrameProcessor.hpp"
#endif

// CUDA kernel
extern "C" void IMP_NV12To444(uint8_t *pNV12Array, uint8_t *pArrayFull, int iWidth, int iHeight, size_t pitchY);

namespace RW{
    namespace IMP{
        namespace COLOR_NV12TORGB{

            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                switch (SubModuleType)
                {
#if defined (CLIENT)
                case CORE::tenSubModule::nenPlayback_Simple:
                {
                    VPL::QT_SIMPLE::tstMyControlStruct *data = static_cast<VPL::QT_SIMPLE::tstMyControlStruct*>(*Data);
                    data->pstBitStream = this->pOutput;

                    //FILE *pFile;
                    //fopen_s(&pFile, "c:\\dummy\\Update.raw", "wb");
                    //fwrite(data->pstBitStream->pBuffer, 1, data->pstBitStream->u32Size, pFile);
                    //fclose(pFile);

                    data->pPayload = this->pPayload;

                    break;
                }
#endif
                default:
                    break;
                }
                cudaFree((void*)this->pInput->cuDevice);
                SAFE_DELETE(this->pInput);
            }

            CORE::tstModuleVersion IMP_ConvColorFramesNV12ToRGB::ModulVersion() {
                CORE::tstModuleVersion version = { 0, 1 };
                return version;
            }

            CORE::tenSubModule IMP_ConvColorFramesNV12ToRGB::SubModulType()
            {
                return CORE::tenSubModule::nenGraphic_ColorNV12ToRGB;
            }


            IMP_ConvColorFramesNV12ToRGB::IMP_ConvColorFramesNV12ToRGB(std::shared_ptr<spdlog::logger> Logger) :
                RW::CORE::AbstractModule(Logger)
            {
                m_u32Height = 0;
                m_u32Width = 0;
            }


            IMP_ConvColorFramesNV12ToRGB::~IMP_ConvColorFramesNV12ToRGB()
            {
            }

            tenStatus IMP_ConvColorFramesNV12ToRGB::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
            {
                m_Logger->debug("Initialise nenGraphic_ColorNV12ToRGB");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
                stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
                if (!data)
                {
                    m_Logger->error("Initialise: Data of stMyControlStruct is empty!");
                    return tenStatus::nenError;
                }
                m_u32Width = data->nWidth;
                m_u32Height = data->nHeight;

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Initialise for nenGraphic_ColorNV12ToRGB module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;

            }

            tenStatus IMP_ConvColorFramesNV12ToRGB::DoRender(CORE::tstControlStruct * ControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;

                m_Logger->debug("DoRender nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
                stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);

                if (!data)
                {
                    m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                    return tenStatus::nenError;
                }
                if (!data->pInput)
                {
                    m_Logger->error("DoRender: pInput is empty!");
                    return tenStatus::nenError;
                }
                if (!data->pInput->cuDevice && !data->pInput->pBitstream)
                        return tenStatus::nenError;

                size_t pitchY;
                uint8_t *arrayY;

                cv::cuda::GpuMat gMat444(m_u32Height, m_u32Width, CV_8UC3);

                cudaError err = cudaMallocPitch((void**)&arrayY, &pitchY, m_u32Width, m_u32Height * 3/2);
                if (err != cudaSuccess) return tenStatus::nenError;

                if (data->pInput->cuDevice){
                    err = cudaMemcpy2D(arrayY, pitchY, (void*)data->pInput->cuDevice, pitchY, m_u32Width, m_u32Height * 3 / 2, cudaMemcpyDeviceToDevice);
                    if (err != cudaSuccess) return tenStatus::nenError;

                    IMP_NV12To444(arrayY, gMat444.data, m_u32Width, m_u32Height, pitchY);
                }
                else if (data->pInput->pBitstream){
                    // --------- not working correctly yet. For DEC\Intel use fourcc MFX_FOURCC_RGB4 instead. ------------
                    printf("IMP_420To444: ConvColorFramesNV12ToRGB not implemented yet for  data->pInput->pBitstream (e.g. Output from DEC_Intel. Use fourcc MFX_FOURCC_RGB4 instead.)");
                    return tenStatus::nenError;

                    //err = cudaMemcpy2D(arrayY, pitchY, (void*)data->pInput->pBitstream->pBuffer, m_u32Width, m_u32Width, m_u32Height * 3 / 2, cudaMemcpyHostToDevice);
                    //if (err != cudaSuccess) return tenStatus::nenError;

                    //IMP_NV12To444(arrayY, gMat444.data, m_u32Width, m_u32Height, pitchY);
                }
                else
                {
                    printf("IMP_420To444: ConvColorFramesNV12ToRGB is not supporting that format!");
                    return tenStatus::nenError;
                }

                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    printf("IMP_420To444: Device synchronize failed! Error = %d\n", err);
                    return tenStatus::nenError;
                }
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("IMP_420To444: kernel() failed to launch error = %d\n", err);
                    return tenStatus::nenError;
                }

                // ------------ Uncomment for checking with Image Watch ---------------------
                //cv::cuda::GpuMat Matrix1(m_u32Height * 3 / 2, m_u32Width, CV_8U, arrayY);
                //cv::Mat matrix1(m_u32Height * 3 / 2, m_u32Width, CV_8U);
                //Matrix1.download(matrix1);
                //cv::Mat matrix2(m_u32Height, m_u32Width, CV_8UC3);
                //gMat444.download(matrix2);

                cv::cuda::cvtColor(gMat444, gMat444, cv::COLOR_YUV2RGB);

                cv::Mat *pMat = new cv::Mat(m_u32Height, m_u32Width, CV_8UC3);
                gMat444.download(*pMat);

                cudaFree(arrayY);

                data->pOutput->pBuffer = pMat->data;
                data->pOutput->u32Size = (uint32_t)(pMat->total() * pMat->channels());

                if (enStatus != tenStatus::nenSuccess || !data->pOutput)
                {
                    m_Logger->error("DoRender: impBase.tensProcessOutput did not succeed!");
                }

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to DoRender for nenGraphic_Color module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }


            tenStatus IMP_ConvColorFramesNV12ToRGB::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenGraphic_ColorNV12ToRGB");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif


#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "DEC_CudaInterop::Deinitialise: Time to Deinitialise for nenDecoder_NVIDIA module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;
            }
        }
    }
}
