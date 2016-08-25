#include "IMP_ConvColorFramesYUV420ToRGB.hpp"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"
#include "opencv2/cudaimgproc.hpp"

#if defined (CLIENT)
#include "..\VPL\QT_simple\VPL_FrameProcessor.hpp"
#endif

// CUDA kernel
extern "C" void IMP_420To444(uint8_t *pYUV420Array, uint8_t *pArrayFull, int iWidth, int iHeight, size_t pitchY);

namespace RW{
    namespace IMP{
        namespace COLOR_YUV420TORGB{

            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                switch (SubModuleType)
                {
#if defined (CLIENT)
                case CORE::tenSubModule::nenPlayback_Simple:
                {
                    VPL::QT_SIMPLE::tstMyControlStruct *data = static_cast<VPL::QT_SIMPLE::tstMyControlStruct*>(*Data);
                    data->pstBitStream = this->pOutput;
                    break;
                }
#endif
                default:
                    break;
                }
            }

            CORE::tstModuleVersion IMP_ConvColorFramesYUV420ToRGB::ModulVersion() {
                CORE::tstModuleVersion version = { 0, 1 };
                return version;
            }

            CORE::tenSubModule IMP_ConvColorFramesYUV420ToRGB::SubModulType()
            {
                return CORE::tenSubModule::nenGraphic_ColorYUV420ToRGB;
            }


            IMP_ConvColorFramesYUV420ToRGB::IMP_ConvColorFramesYUV420ToRGB(std::shared_ptr<spdlog::logger> Logger) :
                RW::CORE::AbstractModule(Logger)
            {
                m_u32Height = 0;
                m_u32Width = 0;
            }


            IMP_ConvColorFramesYUV420ToRGB::~IMP_ConvColorFramesYUV420ToRGB()
            {
            }

            tenStatus IMP_ConvColorFramesYUV420ToRGB::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
            {
                m_Logger->debug("Initialise nenGraphic_ColorYUV420ToRGB");
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
                m_Logger->trace() << "Time to Initialise for nenGraphic_ColorYUV420ToRGB module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;

            }

            tenStatus IMP_ConvColorFramesYUV420ToRGB::DoRender(CORE::tstControlStruct * ControlStruct)
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

                size_t pitchY;
                uint8_t *arrayY;

                cv::cuda::GpuMat gMat444(m_u32Height, m_u32Width, CV_8UC3);

                cudaError err = cudaMallocPitch((void**)&arrayY, &pitchY, m_u32Width, 1);
                if (err != cudaSuccess) return tenStatus::nenError;

                cv::cuda::GpuMat gMat420(m_u32Height * 3 / 2, pitchY, CV_8UC1, (void*)data->pInput);
                cv::Mat mat1(m_u32Height * 3 / 2, pitchY, CV_8UC1);
                gMat420.download(mat1);

                IMP_420To444(gMat420.data, gMat444.data, m_u32Width, m_u32Height, pitchY);

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

                cv::cvtColor(gMat444, gMat444, cv::COLOR_YUV2RGB);

                cv::Mat mat(m_u32Height, m_u32Width, CV_8UC3);
                gMat444.download(mat);

                data->pOutput->pBuffer = mat.data;
                data->pOutput->u32Size = (uint32_t)mat.total()*mat.channels();

                //FILE *pFile;
                //pFile = fopen("c:\\dummy\\dummy.raw", "rb");
                //fwrite(mat.data, 1, mat.total()*mat.channels(), pFile);
                //fclose(pFile);

                cudaFree(arrayY);

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


            tenStatus IMP_ConvColorFramesYUV420ToRGB::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenDecoder_NVIDIA");
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
