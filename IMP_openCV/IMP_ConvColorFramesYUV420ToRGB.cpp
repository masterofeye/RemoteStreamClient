#include "IMP_ConvColorFramesYUV420ToRGB.hpp"
#include "..\VPL_QT\VPL_FrameProcessor.hpp"
#include <cuda_runtime.h>
#include "opencv2/cudev/common.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudev/common.hpp"
#include "opencv2/cudaimgproc.hpp"

// CUDA kernel
extern "C" void IMP_420To444(uint8_t *pArrayYUV420, uint8_t *pArrayFull, int iWidth, int iHeight, size_t pitchY);


namespace RW{
    namespace IMP{
        namespace COLOR_YUV420TORGB{

            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                switch (SubModuleType)
                {
                case CORE::tenSubModule::nenPlayback_Simple:
                {
                    VPL::tstMyControlStruct *data = static_cast<VPL::tstMyControlStruct*>(*Data);
                    data->pstBitStream = this->pOutput;
                    break;
                }
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

                /* Check for previous errors */
                cudaError err;
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("IMP_Base::tensProcessOutput: cudaGetLastError returns error = %d\n", err);
                    return tenStatus::nenError;
                }
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    printf("IMP_Base::tensProessOutput: Device synchronize failed! Error = %d\n", err);
                    return tenStatus::nenError;
                }
                /* Check done */

                size_t pitchY, pitch;
                uint8_t *arrayY;
                uint8_t *array420 = (uint8_t*)data->pInput;

                cv::cuda::GpuMat gMat(m_u32Height * 3, m_u32Width, CV_8UC1);

                err = cudaMallocPitch((void**)&arrayY, &pitchY, m_u32Width, 1);
                if (err != cudaSuccess) return tenStatus::nenError;

                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    printf("IMP_Base::tensProessOutput: Device synchronize failed! Error = %d\n", err);
                    return tenStatus::nenError;
                }
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("IMP_Base::tensProcessOutput: cudaGetLastError returns error = %d\n", err);
                    return tenStatus::nenError;
                }

                IMP_420To444(array420, gMat.data, m_u32Width, m_u32Height, pitchY);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("IMP_420To444: kernel() failed to launch error = %d\n", err);
                    return tenStatus::nenError;
                }
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    printf("IMP_420To444: Device synchronize failed! Error = %d\n", err);
                    return tenStatus::nenError;
                }

				cv::cuda::GpuMat g_mat[3] = { gMat(cv::Rect(0, 0, m_u32Width, m_u32Height)), gMat(cv::Rect(0, m_u32Height, m_u32Width, m_u32Height)), gMat(cv::Rect(0, 2 * m_u32Height, m_u32Width, m_u32Height)) };
				cv::cuda::GpuMat gMat3(m_u32Height, m_u32Width, CV_8UC3, g_mat);

                cv::cuda::cvtColor(gMat3, gMat3, cv::COLOR_YUV2RGB);

                cv::Mat mat(m_u32Height, m_u32Width, CV_8UC3);
                gMat3.download(mat);
                data->pOutput->pBuffer = mat.data;
                data->pOutput->u32Size = mat.total();

                FILE *pFile;
                pFile = fopen("C:\\dummy\\test.raw", "wb");
                fwrite(data->pOutput->pBuffer, 1, data->pOutput->u32Size, pFile);
                fclose(pFile);

                cudaFree(arrayY);
                cudaFree(array420);
                //cudaFree(array444);

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
