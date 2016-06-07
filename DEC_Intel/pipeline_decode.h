/*****************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or
nondisclosure agreement with Intel Corporation and may not be copied
or disclosed except in accordance with the terms of that agreement.
Copyright(c) 2005-2014 Intel Corporation. All Rights Reserved.

*****************************************************************************/

#ifndef __PIPELINE_DECODE_H__
#define __PIPELINE_DECODE_H__

#include "mfx_buffering.h"
#include "mfxmvc.h"
#include "AbstractModule.hpp"

class MFXFrameAllocator;
struct mfxAllocatorParams;
class MFXVideoUSER;
struct MFXPlugin;
class IGFXS3DControl;
class CHWDevice;

template<>struct mfx_ext_buffer_id<mfxExtMVCSeqDesc>{
    enum { id = MFX_EXTBUFF_MVC_SEQ_DESC };
};

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
            //CDecodingPipeline();
            virtual ~CDecodingPipeline();
            void SetEncodedData(tstBitStream *pstEncodedStream)
            {
                m_mfxBS.Data = (mfxU8*)pstEncodedStream->pBuffer;
                m_mfxBS.DataLength = pstEncodedStream->u32Size;
                m_mfxBS.MaxLength = pstEncodedStream->u32Size;
                m_mfxBS.DataFlag = MFX_BITSTREAM_COMPLETE_FRAME;
            }
            tstBitStream *GetOutput(){ return m_pOutput; }

            virtual mfxStatus Init(tstInputParams *pParams);
            virtual mfxStatus RunDecoding(tstBitStream *pPayload);

            virtual void PrintInfo();
            virtual mfxStatus ResetDecoder();
            virtual mfxStatus ResetDevice();

        protected: // functions
            virtual void Close();

            void SetExtBuffersFlag()       { m_bIsExtBuffers = true; }
            virtual mfxStatus InitMfxParams();

            // function for allocating a specific external buffer
            template <typename Buffer>
            mfxStatus AllocateExtBuffer();
            virtual void DeleteExtBuffers();

            virtual mfxStatus AllocateExtMVCBuffers();
            virtual void    DeallocateExtMVCBuffers();

            virtual void AttachExtParam();

            virtual mfxStatus CreateAllocator();
            virtual mfxStatus CreateHWDevice();
            virtual mfxStatus AllocFrames();
            virtual void DeleteFrames();
            virtual void DeleteAllocator();
            mfxStatus InitForFirstFrame();


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
            mfxBitstream            m_mfxBS; // contains encoded data m_mfxBS
            bool                    m_bFirstFrameInitialized;
            tstInputParams          *m_pInputParams;
            RW::tstBitStream        *m_pOutput;

            MFXVideoSession         m_mfxSession;
            MFXVideoDECODE*         m_pmfxDEC;
            mfxVideoParam           m_mfxVideoParams;
            std::auto_ptr<MFXVideoUSER>  m_pUserModule;
            std::auto_ptr<MFXPlugin> m_pPlugin;
            std::vector<mfxExtBuffer *> m_ExtBuffers;

            MFXFrameAllocator*      m_pMFXAllocator;
            mfxAllocatorParams*     m_pmfxAllocatorParams;
            MemType                 m_memType;      // memory type of surfaces to use
            bool                    m_bExternalAlloc; // use memory allocator as external for Media SDK
            mfxFrameAllocResponse   m_mfxResponse; // memory allocation response for decoder

            msdkFrameSurface*       m_pCurrentFreeSurface; // surface detached from free surfaces array
            msdkOutputSurface*      m_pCurrentFreeOutputSurface; // surface detached from free output surfaces array
            msdkOutputSurface*      m_pCurrentOutputSurface; // surface detached from output surfaces array

            MSDKSemaphore*          m_pDeliverOutputSemaphore; // to access to DeliverOutput method
            MSDKEvent*              m_pDeliveredEvent; // to signal when output surfaces will be processed
            mfxStatus               m_error; // error returned by DeliverOutput method
            bool                    m_bStopDeliverLoop;

            bool                    m_bIsExtBuffers; // indicates if external buffers were allocated
            bool                    m_bIsVideoWall; // indicates special mode: decoding will be done in a loop
            bool                    m_bIsCompleteFrame;
            bool                    m_bPrintLatency;

            mfxU32                  m_nTimeout; // enables timeout for video playback, measured in seconds
            mfxU32                  m_nMaxFps; // limit of fps, if isn't specified equal 0.
            mfxU32                  m_nFrames; //limit number of output frames

            std::vector<msdk_tick>  m_vLatency;

            CHWDevice               *m_hwdev;

        private:
            CDecodingPipeline(const CDecodingPipeline&);
            void operator=(const CDecodingPipeline&);
        };
    }
}
#endif // __PIPELINE_DECODE_H__
