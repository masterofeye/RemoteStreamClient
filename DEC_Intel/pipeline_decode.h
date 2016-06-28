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

#include "mfx_buffering.h"
#include "mfxmvc.h"
#include "AbstractModule.hpp"
#include "mfxvideo++.h"
#include "general_allocator.h"
#include "mfxplugin++.h"
#include "plugin_loader.h"
#include "hw_device.h"
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

            void SetEncodedData(tstBitStream *pstEncodedStream)
            {
                m_mfxBS.DataLength = pstEncodedStream->u32Size;
                memmove(m_mfxBS.Data, m_mfxBS.Data + m_mfxBS.DataOffset, m_mfxBS.DataLength);
                m_mfxBS.DataOffset = 0;
                m_mfxBS.Data = (mfxU8*)pstEncodedStream->pBuffer;
                m_mfxBS.MaxLength = pstEncodedStream->u32Size;
                m_mfxBS.DataFlag = MFX_BITSTREAM_COMPLETE_FRAME;//MFX_BITSTREAM_EOS;
            }
            tstBitStream *GetOutput(){ return m_pOutput; }
            void SetOutput(tstBitStream *pOutput){ m_pOutput = pOutput; }

            virtual mfxStatus Init(tstInputParams *pParams);
            virtual mfxStatus RunDecoding(tstBitStream *pPayload);
            virtual void Close();
            virtual mfxStatus ResetDecoder();
            virtual mfxStatus ResetDevice();

            void SetMultiView();
            void SetExtBuffersFlag()       { m_bIsExtBuffers = true; }
            virtual void PrintInfo();

        protected: // functions
            virtual mfxStatus CreateRenderingWindow(tstInputParams *pParams, bool try_s3d);
            virtual mfxStatus InitMfxParams(tstInputParams *pParams);

            // function for allocating a specific external buffer
            template <typename Buffer>
            mfxStatus AllocateExtBuffer();
            virtual void DeleteExtBuffers();

            virtual mfxStatus AllocateExtMVCBuffers();
            virtual void    DeallocateExtMVCBuffers();

            virtual void AttachExtParam();

            virtual mfxStatus InitVppParams();
            virtual mfxStatus AllocAndInitVppFilters();
            virtual bool IsVppRequired(tstInputParams *pParams);

            virtual mfxStatus CreateAllocator();
            virtual mfxStatus CreateHWDevice();
            virtual mfxStatus AllocFrames();
            virtual void DeleteFrames();
            virtual void DeleteAllocator();

            mfxStatus InitForFirstFrame();
            mfxStatus WriteNextFrameToBuffer(mfxFrameSurface1* frame);

            /** \brief Performs SyncOperation on the current output surface with the specified timeout.
             *
             * @return MFX_ERR_NONE Output surface was successfully synced and delivered.
             * @return MFX_ERR_MORE_DATA Array of output surfaces is empty, need to feed decoder.
             * @return MFX_WRN_IN_EXECUTION Specified timeout have elapsed.
             * @return MFX_ERR_UNKNOWN An error has occurred.
             */
            virtual mfxStatus SyncOutputSurface(mfxU32 wait);
            virtual mfxStatus DeliverOutput(mfxFrameSurface1* frame);
            virtual void PrintPerFrameStat(bool force = false);

            virtual mfxStatus DeliverLoop(void);

            static unsigned int MFX_STDCALL DeliverThreadFunc(void* ctx);

        protected: // variables
            std::shared_ptr<spdlog::logger> m_Logger;
            CSmplYUVWriter          m_FileWriter;
            std::auto_ptr<CSmplBitstreamReader>  m_FileReader;
            mfxBitstream            m_mfxBS; // contains encoded data
            tstInputParams          *m_pInputParams;
            RW::tstBitStream        *m_pOutput;

            MFXVideoSession         m_mfxSession;
            mfxIMPL                 m_impl;
            MFXVideoDECODE*         m_pmfxDEC;
            MFXVideoVPP*            m_pmfxVPP;
            mfxVideoParam           m_mfxVideoParams;
            mfxVideoParam           m_mfxVppVideoParams;
            std::auto_ptr<MFXVideoUSER>  m_pUserModule;
            std::auto_ptr<MFXPlugin> m_pPlugin;
            std::vector<mfxExtBuffer *> m_ExtBuffers;

            GeneralAllocator*       m_pGeneralAllocator;
            mfxAllocatorParams*     m_pmfxAllocatorParams;
            MemType                 m_memType;      // memory type of surfaces to use
            bool                    m_bExternalAlloc; // use memory allocator as external for Media SDK
            bool                    m_bDecOutSysmem; // use system memory between Decoder and VPP, if false - video memory
            mfxFrameAllocResponse   m_mfxResponse; // memory allocation response for decoder
            mfxFrameAllocResponse   m_mfxVppResponse;   // memory allocation response for vpp

            msdkFrameSurface*       m_pCurrentFreeSurface; // surface detached from free surfaces array
            msdkFrameSurface*       m_pCurrentFreeVppSurface; // VPP surface detached from free VPP surfaces array
            msdkOutputSurface*      m_pCurrentFreeOutputSurface; // surface detached from free output surfaces array
            msdkOutputSurface*      m_pCurrentOutputSurface; // surface detached from output surfaces array

            MSDKSemaphore*          m_pDeliverOutputSemaphore; // to access to DeliverOutput method
            MSDKEvent*              m_pDeliveredEvent; // to signal when output surfaces will be processed
            mfxStatus               m_error; // error returned by DeliverOutput method
            bool                    m_bStopDeliverLoop;

            eWorkMode               m_eWorkMode; // work mode for the pipeline
            bool                    m_bIsMVC; // enables MVC mode (need to support several files as an output)
            bool                    m_bIsExtBuffers; // indicates if external buffers were allocated
            bool                    m_bIsVideoWall; // indicates special mode: decoding will be done in a loop
            bool                    m_bIsCompleteFrame;
            mfxU32                  m_fourcc; // color format of vpp out, i420 by default
            bool                    m_bPrintLatency;

            mfxU16                  m_vppOutWidth;
            mfxU16                  m_vppOutHeight;

            bool    m_bFirstFrameInitialized;
            
            mfxU32                  m_nTimeout; // enables timeout for video playback, measured in seconds
            mfxU32                  m_nMaxFps; // limit of fps, if isn't specified equal 0.
            mfxU32                  m_nFrames; //limit number of output frames

            mfxU16                  m_diMode;
            bool                    m_bVppIsUsed;
            std::vector<msdk_tick>  m_vLatency;

            mfxExtVPPDoNotUse       m_VppDoNotUse;      // for disabling VPP algorithms
            mfxExtVPPDeinterlacing  m_VppDeinterlacing;
            std::vector<mfxExtBuffer*> m_VppExtParams;

            CHWDevice               *m_hwdev;
#if D3D_SURFACES_SUPPORT
            IGFXS3DControl          *m_pS3DControl;

            CDecodeD3DRender         m_d3dRender;
#endif

            bool                    m_bRenderWin;
            mfxU32                  m_nRenderWinX;
            mfxU32                  m_nRenderWinY;
            mfxU32                  m_nRenderWinW;
            mfxU32                  m_nRenderWinH;

            mfxU32                  m_export_mode;
            mfxI32                  m_monitorType;
#if defined(LIBVA_SUPPORT)
            mfxI32                  m_libvaBackend;
            bool                    m_bPerfMode;
#endif // defined(MFX_LIBVA_SUPPORT)

        private:
            CDecodingPipeline(const CDecodingPipeline&);
            void operator=(const CDecodingPipeline&);
        };
    }
}
