/******************************************************************************\
Copyright (c) 2005-2016, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This sample was distributed or derived from the Intel's Media Samples package.
The original version of this sample may be obtained from https://software.intel.com/en-us/intel-media-server-studio
or https://software.intel.com/en-us/media-client-solutions-support.
\**********************************************************************************/

#pragma once

#if D3D_SURFACES_SUPPORT
#pragma warning(disable : 4201)
#include <d3d9.h>
#include <dxva2api.h>
#endif

#include "hw_device.h"
#include "mfx_buffering.h"
#include "mfxmvc.h"
#include "AbstractModule.hpp"
#include "mfxvideo++.h"
#include "general_allocator.h"
#include "mfxplugin++.h"
#include "plugin_loader.h"
#include "decode_render.h"

template<>struct mfx_ext_buffer_id<mfxExtMVCSeqDesc>{
    enum {id = MFX_EXTBUFF_MVC_SEQ_DESC};
};

struct CPipelineStatistics
{
    CPipelineStatistics():
        m_input_count(0),
        m_output_count(0),
        m_synced_count(0),
        m_tick_overall(0),
        m_tick_fread(0),
        m_tick_fwrite(0),
        m_timer_overall(m_tick_overall)
    {
    }
    virtual ~CPipelineStatistics(){}

    mfxU32 m_input_count;     // number of received incoming packets (frames or bitstreams)
    mfxU32 m_output_count;    // number of delivered outgoing packets (frames or bitstreams)
    mfxU32 m_synced_count;

    msdk_tick m_tick_overall; // overall time passed during processing
    msdk_tick m_tick_fread;   // part of tick_overall: time spent to receive incoming data
    msdk_tick m_tick_fwrite;  // part of tick_overall: time spent to deliver outgoing data

    CAutoTimer m_timer_overall; // timer which corresponds to m_tick_overall

private:
    CPipelineStatistics(const CPipelineStatistics&);
    void operator=(const CPipelineStatistics&);
};

namespace RW{
    namespace DEC{
        namespace INTEL{
            typedef struct stInputParams tstInputParams;
            enum MemType;
            enum eWorkMode;

            class CDecodingPipeline :
                public CBuffering,
                public CPipelineStatistics
            {
            public:
                CDecodingPipeline(std::shared_ptr<spdlog::logger> Logger);
                virtual ~CDecodingPipeline();

                virtual mfxStatus Init(tstInputParams *pParams);
                virtual mfxStatus RunDecoding();
                virtual void Close();
                virtual mfxStatus ResetDecoder();
                virtual mfxStatus ResetDevice();
                tstPayloadMsg *GetPayloadMsg();
                mfxStatus SetEncodedData(tstBitStream *pstEncodedStream);
                RW::tstBitStream *GetOutput(){ return m_pOutput; }

            protected: // functions
                virtual mfxStatus InitMfxParams(tstInputParams *pParams);

                virtual mfxStatus InitVppParams();
                virtual mfxStatus AllocAndInitVppFilters();
                virtual bool IsVppRequired(tstInputParams *pParams);

                virtual mfxStatus CreateAllocator();
                virtual mfxStatus CreateHWDevice();
                virtual mfxStatus AllocFrames();
                virtual void DeleteFrames();
                virtual void DeleteAllocator();

                virtual void PrintInfo();
                mfxStatus WriteNextFrameToBuffer(mfxFrameSurface1* frame);

                /** \brief Performs SyncOperation on the current output surface with the specified timeout.
                 *
                 * @return MFX_ERR_NONE Output surface was successfully synced and delivered.
                 * @return MFX_ERR_MORE_DATA Array of output surfaces is empty, need to feed decoder.
                 * @return MFX_WRN_IN_EXECUTION Specified timeout have elapsed.
                 * @return MFX_ERR_UNKNOWN An error has occurred.
                 */
                virtual mfxStatus SyncOutputSurface(mfxU32 wait);
                virtual void PrintPerFrameStat(bool force = false);
                mfxStatus CDecodingPipeline::FirstFrameInit();


            protected: // variables
                std::shared_ptr<spdlog::logger> m_Logger;
                mfxBitstream            m_mfxBS; // contains encoded data
                tstInputParams         *m_pInputParams;
                RW::tstBitStream       *m_pOutput;

                MFXVideoSession         m_mfxSession;
                mfxIMPL                 m_impl;
                MFXVideoDECODE         *m_pmfxDEC;
                MFXVideoVPP            *m_pmfxVPP;
                mfxVideoParam           m_mfxVideoParams;
                mfxVideoParam           m_mfxVppVideoParams;
                std::auto_ptr<MFXVideoUSER>  m_pUserModule;
                std::auto_ptr<MFXPlugin> m_pPlugin;

                GeneralAllocator       *m_pGeneralAllocator;
                mfxAllocatorParams     *m_pmfxAllocatorParams;
                MemType                 m_memType;      // memory type of surfaces to use
                bool                    m_bExternalAlloc; // use memory allocator as external for Media SDK
                bool                    m_bDecOutSysmem; // use system memory between Decoder and VPP, if false - video memory
                mfxFrameAllocResponse   m_mfxResponse; // memory allocation response for decoder
                mfxFrameAllocResponse   m_mfxVppResponse;   // memory allocation response for vpp

                msdkFrameSurface       *m_pCurrentFreeSurface; // surface detached from free surfaces array
                msdkFrameSurface       *m_pCurrentFreeVppSurface; // VPP surface detached from free VPP surfaces array
                msdkOutputSurface      *m_pCurrentFreeOutputSurface; // surface detached from free output surfaces array
                msdkOutputSurface      *m_pCurrentOutputSurface; // surface detached from output surfaces array

                bool                    m_bIsCompleteFrame;
                mfxU32                  m_fourcc; // color format of vpp out, i420 by default
                bool                    m_bPrintLatency;

                mfxU16                  m_vppOutWidth;
                mfxU16                  m_vppOutHeight;

                bool                    m_bFirstFrameInitialized;

                bool                    m_bVppIsUsed;
                std::vector<msdk_tick>  m_vLatency;

                mfxExtVPPDoNotUse       m_VppDoNotUse;      // for disabling VPP algorithms
                std::vector<mfxExtBuffer*> m_VppExtParams;

                CHWDevice               *m_hwdev;

            private:
                CDecodingPipeline(const CDecodingPipeline&);
                void operator=(const CDecodingPipeline&);
            };
        }
    }
}
