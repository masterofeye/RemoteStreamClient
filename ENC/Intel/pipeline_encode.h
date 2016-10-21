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

#ifndef __PIPELINE_ENCODE_H__
#define __PIPELINE_ENCODE_H__

#ifdef D3D_SURFACES_SUPPORT
#pragma warning(disable : 4201)
#endif

#include "sample_params.h"
#include "base_allocator.h"

#include "mfxvp8.h"
#include "mfxvideo++.h"
#include "mfxplugin++.h"

#include "plugin_loader.h"
#include "ENC_Intel_input.h"

#include "..\RWPluginInterface\AbstractModule.hpp"

class CHWDevice;

namespace RW{
    namespace ENC{
        namespace INTEL{

            struct sTask
            {
                mfxBitstream mfxBS;
                mfxSyncPoint EncSyncP;
                std::list<mfxSyncPoint> DependentVppTasks;
                //CSmplBitstreamWriter *pWriter;

                sTask();
                RW::tstBitStream *GetBitstream();
                mfxStatus Reset();
                mfxStatus Init(mfxU32 nBufferSize);
                mfxStatus Close();
            };

            class CEncTaskPool
            {
            public:
                CEncTaskPool();
                virtual ~CEncTaskPool();

                virtual mfxStatus Init(MFXVideoSession* pmfxSession, mfxU32 nPoolSize, mfxU32 nBufferSize);
                virtual mfxStatus GetFreeTask(sTask **ppTask);
                virtual mfxStatus SynchronizeFirstTask();

                virtual void Close();

                inline RW::tstBitStream *GetBitstream(){ return m_stBitstream; }

            protected:
                RW::tstBitStream *m_stBitstream;
                sTask* m_pTasks;
                mfxU32 m_nPoolSize;
                mfxU32 m_nTaskBufferStart;

                MFXVideoSession* m_pmfxSession;

                virtual mfxU32 GetFreeTaskIndex();
            };

            /* This class implements a pipeline with 2 mfx components: vpp (video preprocessing) and encode */
            class CEncodingPipeline
            {
            public:
                CEncodingPipeline();
                virtual ~CEncodingPipeline();

                virtual mfxStatus Init(sInputParams *pParams);
                virtual mfxStatus Run();
                virtual void Close();
                virtual mfxStatus ResetMFXComponents(sInputParams* pParams);
                virtual mfxStatus ResetDevice();

                virtual void  PrintInfo();

                inline void SetInputData(uint8_t *pYUVarray){ m_pYUVarray = pYUVarray; m_bFrameIsLoaded = false; };
                inline RW::tstBitStream *GetBitstream(){ return m_stBitstream; };

            protected:
                uint8_t *m_pYUVarray;
                RW::tstBitStream *m_stBitstream;

                //std::pair<CSmplBitstreamWriter *,
                //          CSmplBitstreamWriter *> m_FileWriters;
                //CSmplYUVReader m_FileReader;
                CEncTaskPool m_TaskPool;

                MFXVideoSession m_mfxSession;
                MFXVideoENCODE* m_pmfxENC;
                MFXVideoVPP* m_pmfxVPP;
                bool m_bFrameIsLoaded;

                mfxVideoParam m_mfxEncParams;
                mfxVideoParam m_mfxVppParams;

                std::auto_ptr<MFXVideoUSER> m_pUserModule;
                std::auto_ptr<MFXPlugin> m_pPlugin;

                MFXFrameAllocator* m_pMFXAllocator;
                mfxAllocatorParams* m_pmfxAllocatorParams;
                MemType m_memType;
                bool m_bExternalAlloc; // use memory allocator as external for Media SDK

                mfxFrameSurface1* m_pEncSurfaces; // frames array for encoder input (vpp output)
                mfxFrameSurface1* m_pVppSurfaces; // frames array for vpp input
                mfxFrameAllocResponse m_EncResponse;  // memory allocation response for encoder
                mfxFrameAllocResponse m_VppResponse;  // memory allocation response for vpp

                // for disabling VPP algorithms
                mfxExtVPPDoNotUse m_VppDoNotUse;
                // for MVC encoder and VPP configuration
                //mfxExtMVCSeqDesc m_MVCSeqDesc;
                mfxExtCodingOption m_CodingOption;
                // for look ahead BRC configuration
                mfxExtCodingOption2 m_CodingOption2;
                // HEVC
                mfxExtHEVCParam m_ExtHEVCParam;

                // external parameters for each component are stored in a vector
                std::vector<mfxExtBuffer*> m_VppExtParams;
                std::vector<mfxExtBuffer*> m_EncExtParams;

                CHWDevice *m_hwdev;

                virtual mfxStatus InitMfxEncParams(sInputParams *pParams);
                virtual mfxStatus InitMfxVppParams(sInputParams *pParams);

                //virtual mfxStatus InitFileWriters(sInputParams *pParams);
                //virtual void FreeFileWriters();
                //virtual mfxStatus InitFileWriter(CSmplBitstreamWriter **ppWriter, const msdk_char *filename);

                mfxStatus LoadNextYUVFrame(mfxFrameSurface1* pSurface);

                virtual mfxStatus AllocAndInitVppDoNotUse();
                virtual void FreeVppDoNotUse();

                virtual mfxStatus CreateAllocator();
                virtual void DeleteAllocator();

                virtual mfxStatus CreateHWDevice();
                virtual void DeleteHWDevice();

                virtual mfxStatus AllocFrames();
                virtual void DeleteFrames();

                virtual mfxStatus AllocateSufficientBuffer(mfxBitstream* pBS);

                virtual mfxStatus GetFreeTask(sTask **ppTask);
                virtual MFXVideoSession& GetFirstSession(){ return m_mfxSession; }
                virtual MFXVideoENCODE* GetFirstEncoder(){ return m_pmfxENC; }
            };
        }
    }
}
#endif // __PIPELINE_ENCODE_H__
