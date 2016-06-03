/*****************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or
nondisclosure agreement with Intel Corporation and may not be copied
or disclosed except in accordance with the terms of that agreement.
This sample was distributed or derived from the Intel's Media Samples package.
The original version of this sample may be obtained from https://software.intel.com/en-us/intel-media-server-studio
or https://software.intel.com/en-us/media-client-solutions-support.
Copyright(c) 2005-2015 Intel Corporation. All Rights Reserved.

*****************************************************************************/

#pragma once 
//#include "sample_defs.h"

#if D3D_SURFACES_SUPPORT
#pragma warning(disable : 4201)
#include <d3d9.h>
#include <dxva2api.h>
#endif

#include <vector>
#include "hw_device.h"
#include "decode_render.h"
#include "mfx_buffering.h"
#include <memory>

#include "sample_utils.h"
#include "sample_params.h"
#include "base_allocator.h"

#include "mfxmvc.h"
//#include "mfxjpeg.h"
#include "mfxplugin.h"
#include "mfxplugin++.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"

#include "plugin_loader.h"
#include "general_allocator.h"
#include "AbstractModule.hpp"

namespace RW{
    namespace DEC{

        enum MemType {
            SYSTEM_MEMORY = 0x00,
            D3D9_MEMORY = 0x01,
            D3D11_MEMORY = 0x02,
        };

        enum eWorkMode {
            MODE_PERFORMANCE,
            MODE_RENDERING,
            MODE_FILE_DUMP
        };

        typedef struct sInputParams
        {
            mfxU32 videoType; // MFX_CODEC_AVC for h264, MFX_CODEC_HEVC for h265
            eWorkMode mode;
            MemType memType;   // SYSTEM_MEMORY or D3D9_MEMORY or D3D11_MEMORY
            bool    bUseHWLib; // true if application wants to use HW mfx library (platform specific SDK implementation)
            //bool    bIsMVC; // true if Multi-View Codec is in use. Stereoscopic Video Coding 
            bool    bLowLat; // low latency mode
            bool    bCalLat; // latency calculation
            mfxU32  nMaxFPS; //rendering limited by certain fps

            mfxU32  nWallW; //number of windows located in each row
            mfxU32  nWallH; //number of windows located in each column
            mfxU32  nWallCell;    //order of video window in table that will be rendered
            mfxU32  nWallMonitor; //monitor id, 0,1,.. etc
            bool    bWallNoTitle; //whether to show title for each window with fps value
            mfxU32  nWallTimeout; //timeout for -wall option

            mfxU32  numViews; // number of views for Multi-View Codec
            //mfxU32  nRotation; // rotation for Motion JPEG Codec

            mfxU16  nAsyncDepth; // depth of asynchronous pipeline. default value is 4. must be between 1 and 20
            mfxU16  gpuCopy; // GPU Copy mode (three-state option): MFX_GPUCOPY_DEFAULT or MFX_GPUCOPY_ON or MFX_GPUCOPY_OFF

            //MultiThreading only on Win32 or Win64
            mfxU16  nThreadsNum;  //number of mediasdk task threads
            mfxI32  SchedulingType;  //scheduling type of mediasdk task threads
            mfxI32  Priority;  //priority of mediasdk task threads

            mfxU16  scrWidth;  //screen resolution width
            mfxU16  scrHeight;  //screen resolution height

            mfxU16  Width;  //output width
            mfxU16  Height;  //output height

            mfxU32  fourcc;  //Output format parameters: MFX_FOURCC_NV12 or MFX_FOURCC_RGB4 or MFX_FOURCC_P010 or MFX_FOURCC_A2RGB10
            mfxU32  nFrames;
            mfxU16  eDeinterlace;  //enable deinterlacing BOB/ADI; MFX_DEINTERLACING_BOB or MFX_DEINTERLACING_ADVANCED; 

            bool    bRenderWin;

#if defined(LIBVA_SUPPORT) 
            mfxI32  libvaBackend;  //MFX_LIBVA_X11 or MFX_LIBVA_WAYLAND or MFX_LIBVA_DRM_MODESET; only works with memType = D3D9_MEMORY and mode = MODE_RENDERING
            mfxU32  nRenderWinX;
            mfxU32  nRenderWinY;

            mfxI32  monitorType;  //has to be below MFX_MONITOR_MAXNUMBER
            bool    bPerfMode;
#endif // defined(MFX_LIBVA_SUPPORT)

            msdk_char     strSrcFile[MSDK_MAX_FILENAME_LEN];
            msdk_char     strDstFile[MSDK_MAX_FILENAME_LEN];
            sPluginParams pluginParams;

            sInputParams()
            {
                MSDK_ZERO_MEMORY(*this);
            }
        }tstInputParams;

        //template<>struct mfx_ext_buffer_id<mfxExtMVCSeqDesc>{
        //    enum { id = MFX_EXTBUFF_MVC_SEQ_DESC };
        //};

        struct CPipelineStatistics
        {
            CPipelineStatistics() :
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

        class CDecodingPipeline :
            public CBuffering,
            public CPipelineStatistics
        {

        public:
            CDecodingPipeline(std::shared_ptr<spdlog::logger> Logger);
            ~CDecodingPipeline();

            void SetInputParams(sInputParams *pParams){ m_pInputParams = pParams; }
            void SetEncodedData(tstBitStream *pstEncodedStream)
            {
                m_mfxBS.Data = (mfxU8*)pstEncodedStream->pBuffer;
                m_mfxBS.DataLength = pstEncodedStream->u32Size;
                m_mfxBS.MaxLength = pstEncodedStream->u32Size;
                m_mfxBS.DataFlag = MFX_BITSTREAM_COMPLETE_FRAME;
            }
            tstBitStream *GetOutput(){ return m_pOutput; }

            virtual mfxStatus Init();
            virtual mfxStatus RunDecoding(tstBitStream *pPayload);
            virtual mfxStatus ResetDecoder();
            virtual mfxStatus ResetDevice();

            virtual void PrintInfo();

            mfxBitstream            m_mfxBS; // contains encoded data

        private: // functions
            void SetExtBuffersFlag()       { m_bIsExtBuffers = true; }

            //virtual mfxStatus CreateRenderingWindow(sInputParams *pParams, bool try_s3d);
            virtual mfxStatus InitMfxParams();
            virtual mfxStatus InitForFirstFrame();

            virtual mfxStatus InitVppParams();
            virtual mfxStatus AllocAndInitVppFilters();
            virtual bool IsVppRequired();

            virtual mfxStatus CreateAllocator();
            virtual mfxStatus CreateHWDevice();
            virtual mfxStatus AllocFrames();
            virtual void DeleteFrames();

            /** \brief Performs SyncOperation on the current outputsurface  with the specified timeout.
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

        private: // variables
            std::shared_ptr<spdlog::logger> m_Logger;

            //CSmplYUVWriter          m_FileWriter;
            //std::auto_ptr<CSmplBitstreamReader>  m_FileReader;

            tstBitStream           *m_pOutput;

            bool                    m_bFirstFrameInitialized;
            MFXVideoSession         m_mfxSession;
            mfxIMPL                 m_impl;
            MFXVideoDECODE          *m_pmfxDEC;
            MFXVideoVPP             *m_pmfxVPP;
            mfxVideoParam           m_mfxVideoParams;
            mfxVideoParam           m_mfxVppVideoParams;
            //std::auto_ptr<MFXVideoUSER>  m_pUserModule;
            std::auto_ptr<MFXPlugin> m_pPlugin;
            std::vector<mfxExtBuffer*> m_ExtBuffers;
            tstInputParams          *m_pInputParams;
            GeneralAllocator        *m_pGeneralAllocator;
            mfxAllocatorParams      *m_pmfxAllocatorParams;
            MemType                 m_memType;      // memory type of surfaces to use
            bool                    m_bExternalAlloc; // use memory allocator as external for Media SDK
            bool                    m_bDecOutSysmem; // use system memory between Decoder and VPP, if false - video memory
            mfxFrameAllocResponse   m_mfxResponse; // memory allocation response for decoder
            mfxFrameAllocResponse   m_mfxVppResponse;   // memory allocation response for vpp

            msdkFrameSurface        *m_pCurrentFreeSurface; // surface detached from free surfaces array
            msdkFrameSurface        *m_pCurrentFreeVppSurface; // VPP surface detached from free VPP surfaces array
            msdkOutputSurface       *m_pCurrentFreeOutputSurface; // surface detached from free output surfaces array
            msdkOutputSurface       *m_pCurrentOutputSurface; // surface detached from output surfaces array

            MSDKSemaphore           *m_pDeliverOutputSemaphore; // to access to DeliverOutput method
            MSDKEvent               *m_pDeliveredEvent; // to signal when output surfaces will be processed
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
