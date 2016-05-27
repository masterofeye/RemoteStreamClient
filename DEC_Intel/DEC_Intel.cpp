/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement.
This sample was distributed or derived from the Intel's Media Samples package.
The original version of this sample may be obtained from https://software.intel.com/en-us/intel-media-server-studio
or https://software.intel.com/en-us/media-client-solutions-support.
Copyright(c) 2005-2015 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "DEC_Intel.hpp"

namespace RW{
    namespace DEC{
        DEC_Intel::DEC_Intel(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
        {

            }

        DEC_Intel::~DEC_Intel()
        {
        }

        CORE::tstModuleVersion DEC_Intel::ModulVersion() {
            CORE::tstModuleVersion version = { 0, 1 };
            return version;
        }

        CORE::tenSubModule DEC_Intel::SubModulType()
        {
            return CORE::tenSubModule::nenDecoder_INTEL;
        }

        tenStatus DEC_Intel::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
        {

            tenStatus enStatus = tenStatus::nenSuccess;
            m_Logger->debug("Initialise nenDecoder_INTEL");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

            stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

            if (data == NULL)
            {
                m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
                enStatus = tenStatus::nenError;
                return enStatus;
            }

            mfxStatus sts = MFX_ERR_NONE; // return value check

            m_pParams = &data->inputParams;
            m_pParams->memType = D3D11_MEMORY;

            if (!IsDecodeCodecSupported(m_pParams->videoType))
            {
                m_Logger->error("Unsupported codec");
                sts = MFX_ERR_UNSUPPORTED;
            }
            if (m_pParams->videoType == CODEC_MVC)
            {
                m_pParams->videoType = MFX_CODEC_AVC;
                m_pParams->bIsMVC = true;
            }
            if (m_pParams->mode == MODE_RENDERING)
            {
                if (m_pParams->memType == SYSTEM_MEMORY)
                {
                    m_pParams->memType = D3D9_MEMORY;
                }
            }

            m_pParams->bWallNoTitle = false;

            if (m_pParams->bLowLat || m_pParams->bCalLat)
            {
                if ( !(m_pParams->videoType == MFX_CODEC_HEVC
                    || m_pParams->videoType == MFX_CODEC_AVC
                    || m_pParams->videoType == MFX_CODEC_JPEG) 
                    || m_pParams->bIsMVC)
                {
                    m_Logger->error("low latency || calc latency mode is supported only for H.264 and JPEG codecs");
                    sts = MFX_ERR_UNSUPPORTED;
                }
            }
            if (m_pParams->videoType == MFX_CODEC_JPEG)
            {
                if ((m_pParams->nRotation != 0) &&(m_pParams->nRotation != 90) && (m_pParams->nRotation != 180) && (m_pParams->nRotation != 270))
                {
                    m_Logger->error("jpeg rotate is supported only for 0, 90, 180 and 270 angles");
                    sts = MFX_ERR_UNSUPPORTED;
                }
            }
            
            if (m_pParams->bIsMVC)
                m_Pipeline.SetMultiView();

            sts = m_Pipeline.Init(m_pParams);

            // print stream info
            m_Pipeline.PrintInfo();

            if (sts != MFX_ERR_NONE)
            {
                enStatus = tenStatus::nenError;
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to Initialise for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
        }

        tenStatus DEC_Intel::DoRender(CORE::tstControlStruct * ControlStruct)
        {
            tenStatus enStatus = tenStatus::nenSuccess;

            m_Logger->debug("DoRender nenDecoder_INTEL");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
            stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
            if (data == NULL)
            {
                m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                enStatus = tenStatus::nenError;
                return enStatus;
            }

            mfxStatus sts = MFX_ERR_NONE; // return value check
            sts = m_Pipeline.RunDecoding();

            if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM == sts || MFX_ERR_DEVICE_LOST == sts || MFX_ERR_DEVICE_FAILED == sts)
            {
                if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM == sts)
                {
                    m_Logger->error("Incompatible video parameters detected. Recovering...");
                }
                else
                {
                    m_Logger->error("Hardware device was lost or returned unexpected error. Recovering...");
                    sts = m_Pipeline.ResetDevice();
                }

                sts = m_Pipeline.ResetDecoder(m_pParams);
            }
            if (sts != MFX_ERR_NONE)
            {
                enStatus = tenStatus::nenError;
            }

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to DoRender for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
        }

        tenStatus DEC_Intel::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            m_Logger->debug("Deinitialise nenDecoder_INTEL");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Time to Deinitialise for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
        }

    }

}


