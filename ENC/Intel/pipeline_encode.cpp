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

#include "mfx_samples_config.h"

#include "pipeline_encode.h"
#include "sysmem_allocator.h"

#if D3D_SURFACES_SUPPORT
#include "d3d_allocator.h"
#include "d3d11_allocator.h"

#include "d3d_device.h"
#include "d3d11_device.h"
#endif

#include "hw_device.h"

#include "sample_utils.h"
#include "sample_params.h"


#ifdef LIBVA_SUPPORT
#include "vaapi_allocator.h"
#include "vaapi_device.h"
#endif

namespace RW{
    namespace ENC{
        namespace INTEL{

            CEncTaskPool::CEncTaskPool()
            {
                m_pTasks = NULL;
                m_pmfxSession = NULL;
                m_nTaskBufferStart = 0;
                m_nPoolSize = 0;
            }

            CEncTaskPool::~CEncTaskPool()
            {
                Close();
            }

            mfxStatus CEncTaskPool::Init(MFXVideoSession* pmfxSession, mfxU32 nPoolSize, mfxU32 nBufferSize)
            {
                MSDK_CHECK_POINTER(pmfxSession, MFX_ERR_NULL_PTR);

                MSDK_CHECK_ERROR(nPoolSize, 0, MFX_ERR_UNDEFINED_BEHAVIOR);
                MSDK_CHECK_ERROR(nBufferSize, 0, MFX_ERR_UNDEFINED_BEHAVIOR);

                // nPoolSize must be even in case of 2 output bitstreams
                if (0 != nPoolSize % 2)
                    return MFX_ERR_UNDEFINED_BEHAVIOR;

                m_pmfxSession = pmfxSession;
                m_nPoolSize = nPoolSize;

                m_pTasks = new sTask[m_nPoolSize];
                MSDK_CHECK_POINTER(m_pTasks, MFX_ERR_MEMORY_ALLOC);

                mfxStatus sts = MFX_ERR_NONE;

                for (mfxU32 i = 0; i < m_nPoolSize; i++)
                {
                    sts = m_pTasks[i].Init(nBufferSize);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                }

                return MFX_ERR_NONE;
            }

            mfxStatus CEncTaskPool::SynchronizeFirstTask()
            {
                MSDK_CHECK_POINTER(m_pTasks, MFX_ERR_NOT_INITIALIZED);
                MSDK_CHECK_POINTER(m_pmfxSession, MFX_ERR_NOT_INITIALIZED);

                mfxStatus sts = MFX_ERR_NONE;

                // non-null sync point indicates that task is in execution
                if (NULL != m_pTasks[m_nTaskBufferStart].EncSyncP)
                {
                    sts = m_pmfxSession->SyncOperation(m_pTasks[m_nTaskBufferStart].EncSyncP, MSDK_WAIT_INTERVAL);

                    if (MFX_ERR_NONE == sts)
                    {
                        m_stBitstream = m_pTasks[m_nTaskBufferStart].GetBitstream();
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                        sts = m_pTasks[m_nTaskBufferStart].Reset();
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                        // move task buffer start to the next executing task
                        // the first transform frame to the right with non zero sync point
                        for (mfxU32 i = 0; i < m_nPoolSize; i++)
                        {
                            m_nTaskBufferStart = (m_nTaskBufferStart + 1) % m_nPoolSize;
                            if (NULL != m_pTasks[m_nTaskBufferStart].EncSyncP)
                            {
                                break;
                            }
                        }
                    }
                    else if (MFX_ERR_ABORTED == sts)
                    {
                        while (!m_pTasks[m_nTaskBufferStart].DependentVppTasks.empty())
                        {
                            // find out if the error occurred in a VPP task to perform recovery procedure if applicable
                            sts = m_pmfxSession->SyncOperation(*m_pTasks[m_nTaskBufferStart].DependentVppTasks.begin(), 0);

                            if (MFX_ERR_NONE == sts)
                            {
                                m_pTasks[m_nTaskBufferStart].DependentVppTasks.pop_front();
                                sts = MFX_ERR_ABORTED; // save the status of the encode task
                                continue; // go to next vpp task
                            }
                            else
                            {
                                break;
                            }
                        }
                    }

                    return sts;
                }
                else
                {
                    sts = MFX_ERR_NOT_FOUND; // no tasks left in task buffer
                }
                return sts;
            }

            mfxU32 CEncTaskPool::GetFreeTaskIndex()
            {
                mfxU32 off = 0;

                if (m_pTasks)
                {
                    for (off = 0; off < m_nPoolSize; off++)
                    {
                        if (NULL == m_pTasks[(m_nTaskBufferStart + off) % m_nPoolSize].EncSyncP)
                        {
                            break;
                        }
                    }
                }

                if (off >= m_nPoolSize)
                    return m_nPoolSize;

                return (m_nTaskBufferStart + off) % m_nPoolSize;
            }

            mfxStatus CEncTaskPool::GetFreeTask(sTask **ppTask)
            {
                MSDK_CHECK_POINTER(ppTask, MFX_ERR_NULL_PTR);
                MSDK_CHECK_POINTER(m_pTasks, MFX_ERR_NOT_INITIALIZED);

                mfxU32 index = GetFreeTaskIndex();

                if (index >= m_nPoolSize)
                {
                    return MFX_ERR_NOT_FOUND;
                }

                // return the address of the task
                *ppTask = &m_pTasks[index];

                return MFX_ERR_NONE;
            }

            void CEncTaskPool::Close()
            {
                if (m_pTasks)
                {
                    for (mfxU32 i = 0; i < m_nPoolSize; i++)
                    {
                        m_pTasks[i].Close();
                    }
                }

                MSDK_SAFE_DELETE_ARRAY(m_pTasks);

                m_pmfxSession = NULL;
                m_nTaskBufferStart = 0;
                m_nPoolSize = 0;
            }

            sTask::sTask()
                : EncSyncP(0)
            {
                MSDK_ZERO_MEMORY(mfxBS);
            }

            mfxStatus sTask::Init(mfxU32 nBufferSize)
            {
                Close();

                //pWriter = pwriter;

                mfxStatus sts = Reset();
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                sts = InitMfxBitstream(&mfxBS, nBufferSize);
                MSDK_CHECK_RESULT_SAFE(sts, MFX_ERR_NONE, sts, WipeMfxBitstream(&mfxBS));

                return sts;
            }

            mfxStatus sTask::Close()
            {
                WipeMfxBitstream(&mfxBS);
                EncSyncP = 0;
                DependentVppTasks.clear();

                return MFX_ERR_NONE;
            }

            RW::stBitStream *sTask::GetBitstream()
            {
                RW::tstBitStream *bitStream = new RW::tstBitStream;
                bitStream->u32Size = mfxBS.DataLength;
                bitStream->pBuffer = new uint8_t[bitStream->u32Size];
                memcpy(bitStream->pBuffer, mfxBS.Data, bitStream->u32Size);

                return bitStream;//pWriter->WriteNextFrame(&mfxBS);
            }

            mfxStatus sTask::Reset()
            {
                // mark sync point as free
                EncSyncP = NULL;

                // prepare bit stream
                mfxBS.DataOffset = 0;
                mfxBS.DataLength = 0;

                DependentVppTasks.clear();

                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::AllocAndInitVppDoNotUse()
            {
                m_VppDoNotUse.NumAlg = 4;

                m_VppDoNotUse.AlgList = new mfxU32[m_VppDoNotUse.NumAlg];
                MSDK_CHECK_POINTER(m_VppDoNotUse.AlgList, MFX_ERR_MEMORY_ALLOC);

                m_VppDoNotUse.AlgList[0] = MFX_EXTBUFF_VPP_DENOISE; // turn off denoising (on by default)
                m_VppDoNotUse.AlgList[1] = MFX_EXTBUFF_VPP_SCENE_ANALYSIS; // turn off scene analysis (on by default)
                m_VppDoNotUse.AlgList[2] = MFX_EXTBUFF_VPP_DETAIL; // turn off detail enhancement (on by default)
                m_VppDoNotUse.AlgList[3] = MFX_EXTBUFF_VPP_PROCAMP; // turn off processing amplified (on by default)

                return MFX_ERR_NONE;

            } // CEncodingPipeline::AllocAndInitVppDoNotUse()

            void CEncodingPipeline::FreeVppDoNotUse()
            {
                MSDK_SAFE_DELETE_ARRAY(m_VppDoNotUse.AlgList);
            }

            mfxStatus CEncodingPipeline::InitMfxEncParams(sInputParams *pInParams)
            {
                m_mfxEncParams.mfx.CodecId = pInParams->CodecId;
                m_mfxEncParams.mfx.TargetUsage = pInParams->nTargetUsage; // trade-off between quality and speed
                m_mfxEncParams.mfx.TargetKbps = pInParams->nBitRate; // in Kbps
                m_mfxEncParams.mfx.RateControlMethod = pInParams->nRateControlMethod;
                m_mfxEncParams.mfx.GopRefDist = pInParams->nGopRefDist;
                m_mfxEncParams.mfx.GopPicSize = pInParams->nGopPicSize;
                m_mfxEncParams.mfx.NumRefFrame = pInParams->nNumRefFrame > 0 ? pInParams->nNumRefFrame : 1;
                m_mfxEncParams.mfx.IdrInterval = pInParams->nIdrInterval;

                if (m_mfxEncParams.mfx.RateControlMethod == MFX_RATECONTROL_CQP)
                {
                    m_mfxEncParams.mfx.QPI = pInParams->nQPI;
                    m_mfxEncParams.mfx.QPP = pInParams->nQPP;
                    m_mfxEncParams.mfx.QPB = pInParams->nQPB;
                }
                m_mfxEncParams.mfx.NumSlice = 1;
                ConvertFrameRate(pInParams->dFrameRate, &m_mfxEncParams.mfx.FrameInfo.FrameRateExtN, &m_mfxEncParams.mfx.FrameInfo.FrameRateExtD);
                m_mfxEncParams.mfx.EncodedOrder = 0; // binary flag, 0 signals encoder to take frames in display order


                // specify memory type
                if (D3D9_MEMORY == pInParams->memType || D3D11_MEMORY == pInParams->memType)
                {
                    m_mfxEncParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY;
                }
                else
                {
                    m_mfxEncParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
                }

                // frame info parameters
                m_mfxEncParams.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
                m_mfxEncParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
                m_mfxEncParams.mfx.FrameInfo.PicStruct = pInParams->nPicStruct;

                // set frame size and crops
                if (pInParams->CodecId == MFX_CODEC_HEVC && !memcmp(pInParams->pluginParams.pluginGuid.Data, MFX_PLUGINID_HEVCE_HW.Data, sizeof(MFX_PLUGINID_HEVCE_HW.Data)))
                {
                    // In case of HW HEVC decoder width and height must be aligned to 32 pixels. This limitation is planned to be removed in later versions of plugin
                    m_mfxEncParams.mfx.FrameInfo.Width = MSDK_ALIGN32(pInParams->nDstWidth);
                    m_mfxEncParams.mfx.FrameInfo.Height = MSDK_ALIGN32(pInParams->nDstHeight);
                }
                else
                {
                    // width must be a multiple of 16
                    // height must be a multiple of 16 in case of frame picture and a multiple of 32 in case of field picture
                    m_mfxEncParams.mfx.FrameInfo.Width = MSDK_ALIGN16(pInParams->nDstWidth);
                    m_mfxEncParams.mfx.FrameInfo.Height = (MFX_PICSTRUCT_PROGRESSIVE == m_mfxEncParams.mfx.FrameInfo.PicStruct) ?
                        MSDK_ALIGN16(pInParams->nDstHeight) : MSDK_ALIGN32(pInParams->nDstHeight);
                }

                m_mfxEncParams.mfx.FrameInfo.CropX = 0;
                m_mfxEncParams.mfx.FrameInfo.CropY = 0;
                m_mfxEncParams.mfx.FrameInfo.CropW = pInParams->nDstWidth;
                m_mfxEncParams.mfx.FrameInfo.CropH = pInParams->nDstHeight;

                // configure the depth of the look ahead BRC if specified in command line
                if (pInParams->nLADepth || pInParams->nMaxSliceSize || pInParams->nBRefType)
                {
                    m_CodingOption2.LookAheadDepth = pInParams->nLADepth;
                    m_CodingOption2.MaxSliceSize = pInParams->nMaxSliceSize;
                    m_CodingOption2.BRefType = pInParams->nBRefType;
                    m_EncExtParams.push_back((mfxExtBuffer *)&m_CodingOption2);
                }


                // In case of HEVC when height and/or width divided with 8 but not divided with 16
                // add extended parameter to increase performance
                if ((!((m_mfxEncParams.mfx.FrameInfo.CropW & 15) ^ 8) ||
                    !((m_mfxEncParams.mfx.FrameInfo.CropH & 15) ^ 8)) &&
                    (m_mfxEncParams.mfx.CodecId == MFX_CODEC_HEVC))
                {
                    m_ExtHEVCParam.PicWidthInLumaSamples = m_mfxEncParams.mfx.FrameInfo.CropW;
                    m_ExtHEVCParam.PicHeightInLumaSamples = m_mfxEncParams.mfx.FrameInfo.CropH;
                    m_EncExtParams.push_back((mfxExtBuffer*)&m_ExtHEVCParam);
                }

                if (!m_EncExtParams.empty())
                {
                    m_mfxEncParams.ExtParam = &m_EncExtParams[0]; // vector is stored linearly in memory
                    m_mfxEncParams.NumExtParam = (mfxU16)m_EncExtParams.size();
                }

                m_mfxEncParams.AsyncDepth = pInParams->nAsyncDepth;

                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::InitMfxVppParams(sInputParams *pInParams)
            {
                MSDK_CHECK_POINTER(pInParams, MFX_ERR_NULL_PTR);

                // specify memory type
                if (D3D9_MEMORY == pInParams->memType || D3D11_MEMORY == pInParams->memType)
                {
                    m_mfxVppParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
                }
                else
                {
                    m_mfxVppParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
                }

                // input frame info
                m_mfxVppParams.vpp.In.FourCC = MFX_FOURCC_NV12;
                m_mfxVppParams.vpp.In.PicStruct = pInParams->nPicStruct;;
                ConvertFrameRate(pInParams->dFrameRate, &m_mfxVppParams.vpp.In.FrameRateExtN, &m_mfxVppParams.vpp.In.FrameRateExtD);

                // width must be a multiple of 16
                // height must be a multiple of 16 in case of frame picture and a multiple of 32 in case of field picture
                m_mfxVppParams.vpp.In.Width = MSDK_ALIGN16(pInParams->nWidth);
                m_mfxVppParams.vpp.In.Height = (MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.In.PicStruct) ?
                    MSDK_ALIGN16(pInParams->nHeight) : MSDK_ALIGN32(pInParams->nHeight);

                // set crops in input mfxFrameInfo for correct work of file reader
                // VPP itself ignores crops at initialization
                m_mfxVppParams.vpp.In.CropW = pInParams->nWidth;
                m_mfxVppParams.vpp.In.CropH = pInParams->nHeight;

                // fill output frame info
                MSDK_MEMCPY_VAR(m_mfxVppParams.vpp.Out, &m_mfxVppParams.vpp.In, sizeof(mfxFrameInfo));

                // only resizing is supported
                m_mfxVppParams.vpp.Out.Width = MSDK_ALIGN16(pInParams->nDstWidth);
                m_mfxVppParams.vpp.Out.Height = (MFX_PICSTRUCT_PROGRESSIVE == m_mfxVppParams.vpp.Out.PicStruct) ?
                    MSDK_ALIGN16(pInParams->nDstHeight) : MSDK_ALIGN32(pInParams->nDstHeight);

                // configure and attach external parameters
                AllocAndInitVppDoNotUse();
                m_VppExtParams.push_back((mfxExtBuffer *)&m_VppDoNotUse);

                m_mfxVppParams.ExtParam = &m_VppExtParams[0]; // vector is stored linearly in memory
                m_mfxVppParams.NumExtParam = (mfxU16)m_VppExtParams.size();

                m_mfxVppParams.AsyncDepth = pInParams->nAsyncDepth;

                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::CreateHWDevice()
            {
                mfxStatus sts = MFX_ERR_NONE;
#if D3D_SURFACES_SUPPORT
#if MFX_D3D11_SUPPORT
                if (D3D11_MEMORY == m_memType)
                    m_hwdev = new CD3D11Device();
                else
#endif // #if MFX_D3D11_SUPPORT
                    m_hwdev = new CD3D9Device();

                if (NULL == m_hwdev)
                    return MFX_ERR_MEMORY_ALLOC;

                sts = m_hwdev->Init(
                    NULL,
                    0,
                    MSDKAdapter::GetNumber(GetFirstSession()));
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

#elif LIBVA_SUPPORT
                m_hwdev = CreateVAAPIDevice();
                if (NULL == m_hwdev)
                {
                    return MFX_ERR_MEMORY_ALLOC;
                }
                sts = m_hwdev->Init(NULL, 0, MSDKAdapter::GetNumber(GetFirstSession()));
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
#endif
                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::ResetDevice()
            {
                if (D3D9_MEMORY == m_memType || D3D11_MEMORY == m_memType)
                {
                    return m_hwdev->Reset();
                }
                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::AllocFrames()
            {
                MSDK_CHECK_POINTER(GetFirstEncoder(), MFX_ERR_NOT_INITIALIZED);

                mfxStatus sts = MFX_ERR_NONE;
                mfxFrameAllocRequest EncRequest;
                mfxFrameAllocRequest VppRequest[2];

                mfxU16 nEncSurfNum = 0; // number of surfaces for encoder
                mfxU16 nVppSurfNum = 0; // number of surfaces for vpp

                MSDK_ZERO_MEMORY(EncRequest);
                MSDK_ZERO_MEMORY(VppRequest[0]);
                MSDK_ZERO_MEMORY(VppRequest[1]);

                // Calculate the number of surfaces for components.
                // QueryIOSurf functions tell how many surfaces are required to produce at least 1 output.
                // To achieve better performance we provide extra surfaces.
                // 1 extra surface at input allows to get 1 extra output.

                sts = GetFirstEncoder()->QueryIOSurf(&m_mfxEncParams, &EncRequest);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                if (EncRequest.NumFrameSuggested < m_mfxEncParams.AsyncDepth)
                    return MFX_ERR_MEMORY_ALLOC;

                // The number of surfaces shared by vpp output and encode input.
                nEncSurfNum = EncRequest.NumFrameSuggested;

                if (m_pmfxVPP)
                {
                    // VppRequest[0] for input frames request, VppRequest[1] for output frames request
                    sts = m_pmfxVPP->QueryIOSurf(&m_mfxVppParams, VppRequest);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    // The number of surfaces for vpp input - so that vpp can work at async depth = m_nAsyncDepth
                    nVppSurfNum = VppRequest[0].NumFrameSuggested;
                    // If surfaces are shared by 2 components, c1 and c2. NumSurf = c1_out + c2_in - AsyncDepth + 1
                    nEncSurfNum += nVppSurfNum - m_mfxEncParams.AsyncDepth + 1;
                }

                // prepare allocation requests
                EncRequest.NumFrameSuggested = EncRequest.NumFrameMin = nEncSurfNum;
                MSDK_MEMCPY_VAR(EncRequest.Info, &(m_mfxEncParams.mfx.FrameInfo), sizeof(mfxFrameInfo));
                if (m_pmfxVPP)
                {
                    EncRequest.Type |= MFX_MEMTYPE_FROM_VPPOUT; // surfaces are shared between vpp output and encode input
                }

                // alloc frames for encoder
                sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &EncRequest, &m_EncResponse);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // alloc frames for vpp if vpp is enabled
                if (m_pmfxVPP)
                {
                    VppRequest[0].NumFrameSuggested = VppRequest[0].NumFrameMin = nVppSurfNum;
                    MSDK_MEMCPY_VAR(VppRequest[0].Info, &(m_mfxVppParams.mfx.FrameInfo), sizeof(mfxFrameInfo));

                    sts = m_pMFXAllocator->Alloc(m_pMFXAllocator->pthis, &(VppRequest[0]), &m_VppResponse);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                }

                // prepare mfxFrameSurface1 array for encoder
                m_pEncSurfaces = new mfxFrameSurface1[m_EncResponse.NumFrameActual];
                MSDK_CHECK_POINTER(m_pEncSurfaces, MFX_ERR_MEMORY_ALLOC);

                for (int i = 0; i < m_EncResponse.NumFrameActual; i++)
                {
                    memset(&(m_pEncSurfaces[i]), 0, sizeof(mfxFrameSurface1));
                    MSDK_MEMCPY_VAR(m_pEncSurfaces[i].Info, &(m_mfxEncParams.mfx.FrameInfo), sizeof(mfxFrameInfo));

                    if (m_bExternalAlloc)
                    {
                        m_pEncSurfaces[i].Data.MemId = m_EncResponse.mids[i];
                    }
                    else
                    {
                        // get YUV pointers
                        sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_EncResponse.mids[i], &(m_pEncSurfaces[i].Data));
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                    }
                }

                // prepare mfxFrameSurface1 array for vpp if vpp is enabled
                if (m_pmfxVPP)
                {
                    m_pVppSurfaces = new mfxFrameSurface1[m_VppResponse.NumFrameActual];
                    MSDK_CHECK_POINTER(m_pVppSurfaces, MFX_ERR_MEMORY_ALLOC);

                    for (int i = 0; i < m_VppResponse.NumFrameActual; i++)
                    {
                        MSDK_ZERO_MEMORY(m_pVppSurfaces[i]);
                        MSDK_MEMCPY_VAR(m_pVppSurfaces[i].Info, &(m_mfxVppParams.mfx.FrameInfo), sizeof(mfxFrameInfo));

                        if (m_bExternalAlloc)
                        {
                            m_pVppSurfaces[i].Data.MemId = m_VppResponse.mids[i];
                        }
                        else
                        {
                            sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, m_VppResponse.mids[i], &(m_pVppSurfaces[i].Data));
                            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                        }
                    }
                }

                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::CreateAllocator()
            {
                mfxStatus sts = MFX_ERR_NONE;

                if (D3D9_MEMORY == m_memType || D3D11_MEMORY == m_memType)
                {
#if D3D_SURFACES_SUPPORT
                    sts = CreateHWDevice();
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    mfxHDL hdl = NULL;
                    mfxHandleType hdl_t =
#if MFX_D3D11_SUPPORT
                        D3D11_MEMORY == m_memType ? MFX_HANDLE_D3D11_DEVICE :
#endif // #if MFX_D3D11_SUPPORT
                        MFX_HANDLE_D3D9_DEVICE_MANAGER;

                    sts = m_hwdev->GetHandle(hdl_t, &hdl);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    // handle is needed for HW library only
                    mfxIMPL impl = 0;
                    m_mfxSession.QueryIMPL(&impl);
                    if (impl != MFX_IMPL_SOFTWARE)
                    {
                        sts = m_mfxSession.SetHandle(hdl_t, hdl);
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                    }

                    // create D3D allocator
#if MFX_D3D11_SUPPORT
                    if (D3D11_MEMORY == m_memType)
                    {
                        m_pMFXAllocator = new D3D11FrameAllocator;
                        MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

                        D3D11AllocatorParams *pd3dAllocParams = new D3D11AllocatorParams;
                        MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
                        pd3dAllocParams->pDevice = reinterpret_cast<ID3D11Device *>(hdl);

                        m_pmfxAllocatorParams = pd3dAllocParams;
                    }
                    else
#endif // #if MFX_D3D11_SUPPORT
                    {
                        m_pMFXAllocator = new D3DFrameAllocator;
                        MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

                        D3DAllocatorParams *pd3dAllocParams = new D3DAllocatorParams;
                        MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
                        pd3dAllocParams->pManager = reinterpret_cast<IDirect3DDeviceManager9 *>(hdl);

                        m_pmfxAllocatorParams = pd3dAllocParams;
                    }

                    /* In case of video memory we must provide MediaSDK with external allocator
                    thus we demonstrate "external allocator" usage model.
                    Call SetAllocator to pass allocator to Media SDK */
                    sts = m_mfxSession.SetFrameAllocator(m_pMFXAllocator);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    m_bExternalAlloc = true;
#endif
#ifdef LIBVA_SUPPORT
                    sts = CreateHWDevice();
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                    /* It's possible to skip failed result here and switch to SW implementation,
                    but we don't process this way */

                    mfxHDL hdl = NULL;
                    sts = m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);
                    // provide device manager to MediaSDK
                    sts = m_mfxSession.SetHandle(MFX_HANDLE_VA_DISPLAY, hdl);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    // create VAAPI allocator
                    m_pMFXAllocator = new vaapiFrameAllocator;
                    MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

                    vaapiAllocatorParams *p_vaapiAllocParams = new vaapiAllocatorParams;
                    MSDK_CHECK_POINTER(p_vaapiAllocParams, MFX_ERR_MEMORY_ALLOC);

                    p_vaapiAllocParams->m_dpy = (VADisplay)hdl;
                    m_pmfxAllocatorParams = p_vaapiAllocParams;

                    /* In case of video memory we must provide MediaSDK with external allocator
                    thus we demonstrate "external allocator" usage model.
                    Call SetAllocator to pass allocator to mediasdk */
                    sts = m_mfxSession.SetFrameAllocator(m_pMFXAllocator);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    m_bExternalAlloc = true;
#endif
                }
                else
                {
#ifdef LIBVA_SUPPORT
                    //in case of system memory allocator we also have to pass MFX_HANDLE_VA_DISPLAY to HW library
                    mfxIMPL impl;
                    m_mfxSession.QueryIMPL(&impl);

                    if(MFX_IMPL_HARDWARE == MFX_IMPL_BASETYPE(impl))
                    {
                        sts = CreateHWDevice();
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                        mfxHDL hdl = NULL;
                        sts = m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);
                        // provide device manager to MediaSDK
                        sts = m_mfxSession.SetHandle(MFX_HANDLE_VA_DISPLAY, hdl);
                        MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                    }
#endif

                    // create system memory allocator
                    m_pMFXAllocator = new SysMemFrameAllocator;
                    MSDK_CHECK_POINTER(m_pMFXAllocator, MFX_ERR_MEMORY_ALLOC);

                    /* In case of system memory we demonstrate "no external allocator" usage model.
                    We don't call SetAllocator, Media SDK uses internal allocator.
                    We use system memory allocator simply as a memory manager for application*/
                }

                // initialize memory allocator
                sts = m_pMFXAllocator->Init(m_pmfxAllocatorParams);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                return MFX_ERR_NONE;
            }

            void CEncodingPipeline::DeleteFrames()
            {
                // delete surfaces array
                MSDK_SAFE_DELETE_ARRAY(m_pEncSurfaces);
                MSDK_SAFE_DELETE_ARRAY(m_pVppSurfaces);

                // delete frames
                if (m_pMFXAllocator)
                {
                    m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_EncResponse);
                    m_pMFXAllocator->Free(m_pMFXAllocator->pthis, &m_VppResponse);
                }
            }

            void CEncodingPipeline::DeleteHWDevice()
            {
                MSDK_SAFE_DELETE(m_hwdev);
            }

            void CEncodingPipeline::DeleteAllocator()
            {
                // delete allocator
                MSDK_SAFE_DELETE(m_pMFXAllocator);
                MSDK_SAFE_DELETE(m_pmfxAllocatorParams);

                DeleteHWDevice();
            }

            CEncodingPipeline::CEncodingPipeline()
            {
                m_pmfxENC = NULL;
                m_pmfxVPP = NULL;
                m_pMFXAllocator = NULL;
                m_pmfxAllocatorParams = NULL;
                m_memType = SYSTEM_MEMORY;
                m_bExternalAlloc = false;
                m_pEncSurfaces = NULL;
                m_pVppSurfaces = NULL;

                MSDK_ZERO_MEMORY(m_VppDoNotUse);
                m_VppDoNotUse.Header.BufferId = MFX_EXTBUFF_VPP_DONOTUSE;
                m_VppDoNotUse.Header.BufferSz = sizeof(m_VppDoNotUse);

                MSDK_ZERO_MEMORY(m_CodingOption);
                m_CodingOption.Header.BufferId = MFX_EXTBUFF_CODING_OPTION;
                m_CodingOption.Header.BufferSz = sizeof(m_CodingOption);

                MSDK_ZERO_MEMORY(m_CodingOption2);
                m_CodingOption2.Header.BufferId = MFX_EXTBUFF_CODING_OPTION2;
                m_CodingOption2.Header.BufferSz = sizeof(m_CodingOption2);

                MSDK_ZERO_MEMORY(m_ExtHEVCParam);
                m_ExtHEVCParam.Header.BufferId = MFX_EXTBUFF_HEVC_PARAM;
                m_ExtHEVCParam.Header.BufferSz = sizeof(m_ExtHEVCParam);

#if D3D_SURFACES_SUPPORT
                m_hwdev = NULL;
#endif

                MSDK_ZERO_MEMORY(m_mfxEncParams);
                MSDK_ZERO_MEMORY(m_mfxVppParams);

                MSDK_ZERO_MEMORY(m_EncResponse);
                MSDK_ZERO_MEMORY(m_VppResponse);
            }

            CEncodingPipeline::~CEncodingPipeline()
            {
                Close();
            }

            mfxStatus CEncodingPipeline::Init(sInputParams *pParams)
            {
                MSDK_CHECK_POINTER(pParams, MFX_ERR_NULL_PTR);

                mfxStatus sts = MFX_ERR_NONE;

                mfxInitParam initPar;
                mfxVersion version;     // real API version with which library is initialized

                MSDK_ZERO_MEMORY(initPar);

                // we set version to 1.0 and later we will query actual version of the library which will got leaded
                initPar.Version.Major = 1;
                initPar.Version.Minor = 0;

                initPar.GPUCopy = pParams->gpuCopy;

                // Init session
                if (pParams->bUseHWLib) {
                    // try searching on all display adapters
                    initPar.Implementation = MFX_IMPL_HARDWARE;

                    // if d3d11 surfaces are used ask the library to run acceleration through D3D11
                    // feature may be unsupported due to OS or MSDK API version
                    if (D3D11_MEMORY == pParams->memType)
                        initPar.Implementation |= MFX_IMPL_VIA_D3D11;

                    sts = m_mfxSession.InitEx(initPar);

                    // MSDK API version may not support multiple adapters - then try initialize on the default
                    if (MFX_ERR_NONE != sts) {
                        initPar.Implementation = (initPar.Implementation & !MFX_IMPL_HARDWARE_ANY) | MFX_IMPL_HARDWARE;
                        sts = m_mfxSession.InitEx(initPar);
                        if (MFX_ERR_NONE != sts){
                            initPar.Implementation = MFX_IMPL_HARDWARE_ANY;
                            sts = m_mfxSession.Init(initPar.Implementation, &initPar.Version);
                            if (MFX_ERR_NONE != sts){
                                pParams->bUseHWLib = false;
                                sts = Init(pParams);
                                return sts;
                            }
                        }
                    }
                }
                else {
                    initPar.Implementation = MFX_IMPL_SOFTWARE;
                    sts = m_mfxSession.InitEx(initPar);
                    if (MFX_ERR_NONE != sts)
                    {
                        initPar.Implementation = MFX_IMPL_AUTO_ANY;
                        sts = m_mfxSession.InitEx(initPar);
                    }
                }

                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                sts = MFXQueryVersion(m_mfxSession, &version); // get real API version of the loaded library
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                if ((pParams->nRateControlMethod == MFX_RATECONTROL_LA) && !CheckVersion(&version, MSDK_FEATURE_LOOK_AHEAD)) {
                    msdk_printf(MSDK_STRING("error: Look ahead is not supported in the %d.%d API version\n"),
                        version.Major, version.Minor);
                    return MFX_ERR_UNSUPPORTED;
                }

                if (CheckVersion(&version, MSDK_FEATURE_PLUGIN_API)) {
                    /* Here we actually define the following codec initialization scheme:
                    *  1. If plugin path or guid is specified: we load user-defined plugin (example: HEVC encoder plugin)
                    *  2. If plugin path not specified:
                    *    2.a) we check if codec is distributed as a mediasdk plugin and load it if yes
                    *    2.b) if codec is not in the list of mediasdk plugins, we assume, that it is supported inside mediasdk library
                    */
                    if (pParams->pluginParams.type == MFX_PLUGINLOAD_TYPE_FILE && strlen(pParams->pluginParams.strPluginPath))
                    {
                        m_pUserModule.reset(new MFXVideoUSER(m_mfxSession));
                        m_pPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, m_mfxSession, pParams->pluginParams.pluginGuid, 1, pParams->pluginParams.strPluginPath, (mfxU32)strlen(pParams->pluginParams.strPluginPath)));
                        if (m_pPlugin.get() == NULL) sts = MFX_ERR_UNSUPPORTED;
                    }
                    else
                    {
                        if (AreGuidsEqual(pParams->pluginParams.pluginGuid, MSDK_PLUGINGUID_NULL))
                        {
                            mfxIMPL impl = pParams->bUseHWLib ? MFX_IMPL_HARDWARE : MFX_IMPL_SOFTWARE;
                            pParams->pluginParams.pluginGuid = msdkGetPluginUID(impl, MSDK_VENCODE, pParams->CodecId);
                        }
                        if (!AreGuidsEqual(pParams->pluginParams.pluginGuid, MSDK_PLUGINGUID_NULL))
                        {
                            m_pPlugin.reset(LoadPlugin(MFX_PLUGINTYPE_VIDEO_ENCODE, m_mfxSession, pParams->pluginParams.pluginGuid, 1));
                            if (m_pPlugin.get() == NULL) sts = MFX_ERR_UNSUPPORTED;
                        }
                        if (sts == MFX_ERR_UNSUPPORTED)
                        {
                            msdk_printf(MSDK_STRING("Default plugin cannot be loaded (possibly you have to define plugin explicitly)\n"));
                        }
                    }
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                }

                // set memory type
                m_memType = pParams->memType;

                // create and init frame allocator
                sts = CreateAllocator();
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                sts = InitMfxEncParams(pParams);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                sts = InitMfxVppParams(pParams);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // create encoder
                m_pmfxENC = new MFXVideoENCODE(m_mfxSession);
                MSDK_CHECK_POINTER(m_pmfxENC, MFX_ERR_MEMORY_ALLOC);

                // create preprocessor if resizing was requested from command line
                // or if different FourCC is set in InitMfxVppParams
                if (pParams->nWidth != pParams->nDstWidth ||
                    pParams->nHeight != pParams->nDstHeight ||
                    m_mfxVppParams.vpp.In.FourCC != m_mfxVppParams.vpp.Out.FourCC)
                {
                    m_pmfxVPP = new MFXVideoVPP(m_mfxSession);
                    MSDK_CHECK_POINTER(m_pmfxVPP, MFX_ERR_MEMORY_ALLOC);
                }

                sts = ResetMFXComponents(pParams);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                return MFX_ERR_NONE;
            }

            void CEncodingPipeline::Close()
            {
                MSDK_SAFE_DELETE(m_pmfxENC);
                MSDK_SAFE_DELETE(m_pmfxVPP);

                FreeVppDoNotUse();

                DeleteFrames();

                m_pPlugin.reset();

                m_TaskPool.Close();
                m_mfxSession.Close();

                // allocator if used as external for MediaSDK must be deleted after SDK components
                DeleteAllocator();
            }

            mfxStatus CEncodingPipeline::ResetMFXComponents(sInputParams* pParams)
            {
                MSDK_CHECK_POINTER(pParams, MFX_ERR_NULL_PTR);
                MSDK_CHECK_POINTER(m_pmfxENC, MFX_ERR_NOT_INITIALIZED);

                mfxStatus sts = MFX_ERR_NONE;

                sts = m_pmfxENC->Close();
                MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                if (m_pmfxVPP)
                {
                    sts = m_pmfxVPP->Close();
                    MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_INITIALIZED);
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                }

                // free allocated frames
                DeleteFrames();

                m_TaskPool.Close();

                sts = AllocFrames();
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                sts = m_pmfxENC->Init(&m_mfxEncParams);
                if (MFX_WRN_PARTIAL_ACCELERATION == sts)
                {
                    msdk_printf(MSDK_STRING("WARNING: partial acceleration\n"));
                    MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
                }

                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                if (m_pmfxVPP)
                {
                    sts = m_pmfxVPP->Init(&m_mfxVppParams);
                    if (MFX_WRN_PARTIAL_ACCELERATION == sts)
                    {
                        msdk_printf(MSDK_STRING("WARNING: partial acceleration\n"));
                        MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
                    }
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                }

                mfxU32 nEncodedDataBufferSize = m_mfxEncParams.mfx.FrameInfo.Width * m_mfxEncParams.mfx.FrameInfo.Height * 4;
                sts = m_TaskPool.Init(&m_mfxSession, m_mfxEncParams.AsyncDepth, nEncodedDataBufferSize);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::AllocateSufficientBuffer(mfxBitstream* pBS)
            {
                MSDK_CHECK_POINTER(pBS, MFX_ERR_NULL_PTR);
                MSDK_CHECK_POINTER(GetFirstEncoder(), MFX_ERR_NOT_INITIALIZED);

                mfxVideoParam par;
                MSDK_ZERO_MEMORY(par);

                // find out the required buffer size
                mfxStatus sts = GetFirstEncoder()->GetVideoParam(&par);
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // reallocate bigger buffer for output
                sts = ExtendMfxBitstream(pBS, par.mfx.BufferSizeInKB * 1000);
                MSDK_CHECK_RESULT_SAFE(sts, MFX_ERR_NONE, sts, WipeMfxBitstream(pBS));

                return MFX_ERR_NONE;
            }

            mfxStatus CEncodingPipeline::GetFreeTask(sTask **ppTask)
            {
                mfxStatus sts = MFX_ERR_NONE;

                sts = m_TaskPool.GetFreeTask(ppTask);
                if (MFX_ERR_NOT_FOUND == sts)
                {
                    sts = m_TaskPool.SynchronizeFirstTask();
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                    //m_stBitstream = m_TaskPool.GetBitstream();
                    //if (!m_stBitstream.pBuffer)
                    //{
                    //    MSDK_PRINT_RET_MSG("CEncodingPipeline::GetFreeTask: Empty Bitstream buffer!");
                    //}

                    // try again
                    sts = m_TaskPool.GetFreeTask(ppTask);
                }

                return sts;
            }

            mfxStatus CEncodingPipeline::LoadNextYUVFrame(mfxFrameSurface1* pSurface)
            {
                if (m_bFrameIsLoaded)
                    return MFX_ERR_MORE_DATA;

                MSDK_CHECK_POINTER(pSurface, MFX_ERR_NULL_PTR);

                mfxU16 w, h, i, pitch;
                mfxU8 *dest, *src;
                mfxFrameInfo& pInfo = pSurface->Info;
                mfxFrameData& pData = pSurface->Data;
				mfxU32 vid = pInfo.FrameId.ViewId;

				if (pInfo.FourCC != MFX_FOURCC_NV12)
					return MFX_ERR_UNSUPPORTED;
				
                if (pInfo.CropH > 0 && pInfo.CropW > 0)
                {
                    w = pInfo.CropW;
                    h = pInfo.CropH;
                }
                else
                {
                    w = pInfo.Width;
                    h = pInfo.Height;
                }
				pitch = pData.Pitch;

				// TODO: Set pData pointers directly; memcpy is not really necessary.

                // read luminance plane
				src = m_pYUVarray;
				dest = pData.Y + pInfo.CropX + pInfo.CropY * pData.Pitch;
				for (i = 0; i < h; i++)
                    memcpy(dest + i * pitch, src + i * w, w);

				// load UV
				src = m_pYUVarray + (mfxU32)pInfo.Height * pitch + pInfo.CropX + (mfxU32)pInfo.CropY * pitch / 2;
				dest = pData.UV + pInfo.CropX + (pInfo.CropY / 2) * pitch;
                for (i = 0; i < h/2; i++)
					memcpy(dest + i * pitch, src + i * w, w);

                m_bFrameIsLoaded = true;
                return MFX_ERR_NONE;
            }


            mfxStatus CEncodingPipeline::Run()
            {
                MSDK_CHECK_POINTER(m_pmfxENC, MFX_ERR_NOT_INITIALIZED);

                mfxStatus sts = MFX_ERR_NONE;

                mfxFrameSurface1* pSurf = NULL; // dispatching pointer

                sTask *pCurrentTask = NULL; // a pointer to the current task
                mfxU16 nEncSurfIdx = 0;     // index of free surface for encoder input (vpp output)
                mfxU16 nVppSurfIdx = 0;     // index of free surface for vpp input

                mfxSyncPoint VppSyncPoint = NULL; // a sync point associated with an asynchronous vpp call
                bool bVppMultipleOutput = false;  // this flag is true if VPP produces more frames at output
                // than consumes at input. E.g. framerate conversion 30 fps -> 60 fps

                sts = MFX_ERR_NONE;

                // main loop, preprocessing and encoding
                while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts)
                {
                    // get a pointer to a free task (bit stream and sync point for encoder)
                    sts = GetFreeTask(&pCurrentTask);
                    MSDK_BREAK_ON_ERROR(sts);

                    // find free surface for encoder input
                    nEncSurfIdx = GetFreeSurface(m_pEncSurfaces, m_EncResponse.NumFrameActual);
                    MSDK_CHECK_ERROR(nEncSurfIdx, MSDK_INVALID_SURF_IDX, MFX_ERR_MEMORY_ALLOC);

                    // point pSurf to encoder surface
                    pSurf = &m_pEncSurfaces[nEncSurfIdx];
                    if (!bVppMultipleOutput)
                    {
                        // if vpp is enabled find free surface for vpp input and point pSurf to vpp surface
                        if (m_pmfxVPP)
                        {
                            nVppSurfIdx = GetFreeSurface(m_pVppSurfaces, m_VppResponse.NumFrameActual);
                            MSDK_CHECK_ERROR(nVppSurfIdx, MSDK_INVALID_SURF_IDX, MFX_ERR_MEMORY_ALLOC);

                            pSurf = &m_pVppSurfaces[nVppSurfIdx];
                        }

                        // load frame from file to surface data
                        // if we share allocator with Media SDK we need to call Lock to access surface data and...
                        if (m_bExternalAlloc)
                        {
                            // get YUV pointers
                            sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, pSurf->Data.MemId, &(pSurf->Data));
                            if (sts == MFX_ERR_LOCK_MEMORY){
                                sts = m_pMFXAllocator->Unlock(m_pMFXAllocator->pthis, pSurf->Data.MemId, &(pSurf->Data));
                                MSDK_BREAK_ON_ERROR(sts);
                                sts = m_pMFXAllocator->Lock(m_pMFXAllocator->pthis, pSurf->Data.MemId, &(pSurf->Data));
                            }
                            MSDK_BREAK_ON_ERROR(sts);
                        }

                        pSurf->Info.FrameId.ViewId = 0;
                        sts = LoadNextYUVFrame(pSurf);
                        MSDK_BREAK_ON_ERROR(sts);

                        // ... after we're done call Unlock
                        if (m_bExternalAlloc)
                        {
                            sts = m_pMFXAllocator->Unlock(m_pMFXAllocator->pthis, pSurf->Data.MemId, &(pSurf->Data));
                            MSDK_BREAK_ON_ERROR(sts);
                        }

                    }

                    // perform preprocessing if required
                    if (m_pmfxVPP)
                    {
                        bVppMultipleOutput = false; // reset the flag before a call to VPP
                        for (;;)
                        {
                            sts = m_pmfxVPP->RunFrameVPPAsync(&m_pVppSurfaces[nVppSurfIdx], &m_pEncSurfaces[nEncSurfIdx],
                                NULL, &VppSyncPoint);

                            if (MFX_ERR_NONE < sts && !VppSyncPoint) // repeat the call if warning and no output
                            {
                                if (MFX_WRN_DEVICE_BUSY == sts)
                                    MSDK_SLEEP(1); // wait if device is busy
                            }
                            else if (MFX_ERR_NONE < sts && VppSyncPoint)
                            {
                                sts = MFX_ERR_NONE; // ignore warnings if output is available
                                break;
                            }
                            else
                                break; // not a warning
                        }

                        // process errors
                        if (MFX_ERR_MORE_DATA == sts)
                        {
                            continue;
                        }
                        else if (MFX_ERR_MORE_SURFACE == sts)
                        {
                            bVppMultipleOutput = true;
                        }
                        else
                        {
                            MSDK_BREAK_ON_ERROR(sts);
                        }
                    }

                    // save the id of preceding vpp task which will produce input data for the encode task
                    if (VppSyncPoint)
                    {
                        pCurrentTask->DependentVppTasks.push_back(VppSyncPoint);
                        VppSyncPoint = NULL;
                    }

                    for (;;)
                    {
                        // at this point surface for encoder contains either a frame from file or a frame processed by vpp
                        sts = m_pmfxENC->EncodeFrameAsync(NULL, &m_pEncSurfaces[nEncSurfIdx], &pCurrentTask->mfxBS, &pCurrentTask->EncSyncP);

                        if (MFX_ERR_NONE < sts && !pCurrentTask->EncSyncP) // repeat the call if warning and no output
                        {
                            if (MFX_WRN_DEVICE_BUSY == sts)
                                MSDK_SLEEP(1); // wait if device is busy
                        }
                        else if (MFX_ERR_NONE < sts && pCurrentTask->EncSyncP)
                        {
                            sts = MFX_ERR_NONE; // ignore warnings if output is available
                            break;
                        }
                        else if (MFX_ERR_NOT_ENOUGH_BUFFER == sts)
                        {
                            sts = AllocateSufficientBuffer(&pCurrentTask->mfxBS);
                            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                        }
                        else
                        {
                            // get next surface and new task for 2nd bitstream in ViewOutput mode
                            MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_BITSTREAM);
                            break;
                        }
                    }
                }

                // means that the input file has ended, need to go to buffering loops
                MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
                // exit in case of other errors
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                if (m_pmfxVPP)
                {
                    // loop to get buffered frames from vpp
                    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts || MFX_ERR_MORE_SURFACE == sts)
                        // MFX_ERR_MORE_SURFACE can be returned only by RunFrameVPPAsync
                        // MFX_ERR_MORE_DATA is accepted only from EncodeFrameAsync
                    {
                        // find free surface for encoder input (vpp output)
                        nEncSurfIdx = GetFreeSurface(m_pEncSurfaces, m_EncResponse.NumFrameActual);
                        MSDK_CHECK_ERROR(nEncSurfIdx, MSDK_INVALID_SURF_IDX, MFX_ERR_MEMORY_ALLOC);

                        for (;;)
                        {
                            sts = m_pmfxVPP->RunFrameVPPAsync(NULL, &m_pEncSurfaces[nEncSurfIdx], NULL, &VppSyncPoint);

                            if (MFX_ERR_NONE < sts && !VppSyncPoint) // repeat the call if warning and no output
                            {
                                if (MFX_WRN_DEVICE_BUSY == sts)
                                    MSDK_SLEEP(1); // wait if device is busy
                            }
                            else if (MFX_ERR_NONE < sts && VppSyncPoint)
                            {
                                sts = MFX_ERR_NONE; // ignore warnings if output is available
                                break;
                            }
                            else
                                break; // not a warning
                        }

                        if (MFX_ERR_MORE_SURFACE == sts)
                        {
                            continue;
                        }
                        else
                        {
                            MSDK_BREAK_ON_ERROR(sts);
                        }

                        // get a free task (bit stream and sync point for encoder)
                        sts = GetFreeTask(&pCurrentTask);
                        MSDK_BREAK_ON_ERROR(sts);

                        // save the id of preceding vpp task which will produce input data for the encode task
                        if (VppSyncPoint)
                        {
                            pCurrentTask->DependentVppTasks.push_back(VppSyncPoint);
                            VppSyncPoint = NULL;
                        }

                        for (;;)
                        {
                            sts = m_pmfxENC->EncodeFrameAsync(NULL, &m_pEncSurfaces[nEncSurfIdx], &pCurrentTask->mfxBS, &pCurrentTask->EncSyncP);

                            if (MFX_ERR_NONE < sts && !pCurrentTask->EncSyncP) // repeat the call if warning and no output
                            {
                                if (MFX_WRN_DEVICE_BUSY == sts)
                                    MSDK_SLEEP(1); // wait if device is busy
                            }
                            else if (MFX_ERR_NONE < sts && pCurrentTask->EncSyncP)
                            {
                                sts = MFX_ERR_NONE; // ignore warnings if output is available
                                break;
                            }
                            else if (MFX_ERR_NOT_ENOUGH_BUFFER == sts)
                            {
                                sts = AllocateSufficientBuffer(&pCurrentTask->mfxBS);
                                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                            }
                            else
                            {
                                // get next surface and new task for 2nd bitstream in ViewOutput mode
                                MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_BITSTREAM);
                                break;
                            }
                        }
                    }

                    // MFX_ERR_MORE_DATA is the correct status to exit buffering loop with
                    // indicates that there are no more buffered frames
                    MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
                    // exit in case of other errors
                    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                }

                // loop to get buffered frames from encoder
                while (MFX_ERR_NONE <= sts)
                {
                    // get a free task (bit stream and sync point for encoder)
                    sts = GetFreeTask(&pCurrentTask);
                    MSDK_BREAK_ON_ERROR(sts);

                    for (;;)
                    {
                        sts = m_pmfxENC->EncodeFrameAsync(NULL, NULL, &pCurrentTask->mfxBS, &pCurrentTask->EncSyncP);

                        if (MFX_ERR_NONE < sts && !pCurrentTask->EncSyncP) // repeat the call if warning and no output
                        {
                            if (MFX_WRN_DEVICE_BUSY == sts)
                                MSDK_SLEEP(1); // wait if device is busy
                        }
                        else if (MFX_ERR_NONE < sts && pCurrentTask->EncSyncP)
                        {
                            sts = MFX_ERR_NONE; // ignore warnings if output is available
                            break;
                        }
                        else if (MFX_ERR_NOT_ENOUGH_BUFFER == sts)
                        {
                            sts = AllocateSufficientBuffer(&pCurrentTask->mfxBS);
                            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
                        }
                        else
                        {
                            // get new task for 2nd bitstream in ViewOutput mode
                            MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_BITSTREAM);
                            break;
                        }
                    }
                    MSDK_BREAK_ON_ERROR(sts);
                }

                // MFX_ERR_MORE_DATA is the correct status to exit buffering loop with
                // indicates that there are no more buffered frames
                MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
                // exit in case of other errors
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                // synchronize all tasks that are left in task pool
                while (MFX_ERR_NONE == sts)
                {
                    sts = m_TaskPool.SynchronizeFirstTask();
                }

                // MFX_ERR_NOT_FOUND is the correct status to exit the loop with
                // EncodeFrameAsync and SyncOperation don't return this status
                MSDK_IGNORE_MFX_STS(sts, MFX_ERR_NOT_FOUND);
                // report any errors that occurred in asynchronous part
                MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

                m_stBitstream = m_TaskPool.GetBitstream();
                MSDK_CHECK_POINTER(m_stBitstream->pBuffer, MFX_ERR_NULL_PTR);

                return sts;
            }

            void CEncodingPipeline::PrintInfo()
            {
				mfxFrameInfo SrcPicInfo = m_mfxVppParams.vpp.In;
				mfxFrameInfo DstPicInfo = m_mfxEncParams.mfx.FrameInfo;

				msdk_printf(MSDK_STRING("Encoding Sample Version %s\n"), MSDK_SAMPLE_VERSION);
				msdk_printf(MSDK_STRING("\nInput file format\t%s\n"), ColorFormatToStr(SrcPicInfo.FourCC));
				//msdk_printf(MSDK_STRING("\nInput file format\t%s\n"), ColorFormatToStr(m_FileReader.m_ColorFormat));
                msdk_printf(MSDK_STRING("Output video\t\t%s\n"), CodecIdToStr(m_mfxEncParams.mfx.CodecId).c_str());

                msdk_printf(MSDK_STRING("Source picture:\n"));
                msdk_printf(MSDK_STRING("\tResolution\t%dx%d\n"), SrcPicInfo.Width, SrcPicInfo.Height);
                msdk_printf(MSDK_STRING("\tCrop X,Y,W,H\t%d,%d,%d,%d\n"), SrcPicInfo.CropX, SrcPicInfo.CropY, SrcPicInfo.CropW, SrcPicInfo.CropH);

                msdk_printf(MSDK_STRING("Destination picture:\n"));
                msdk_printf(MSDK_STRING("\tResolution\t%dx%d\n"), DstPicInfo.Width, DstPicInfo.Height);
                msdk_printf(MSDK_STRING("\tCrop X,Y,W,H\t%d,%d,%d,%d\n"), DstPicInfo.CropX, DstPicInfo.CropY, DstPicInfo.CropW, DstPicInfo.CropH);

                msdk_printf(MSDK_STRING("Frame rate\t%.2f\n"), DstPicInfo.FrameRateExtN * 1.0 / DstPicInfo.FrameRateExtD);
                msdk_printf(MSDK_STRING("Bit rate(Kbps)\t%d\n"), m_mfxEncParams.mfx.TargetKbps);
                msdk_printf(MSDK_STRING("Gop size\t%d\n"), m_mfxEncParams.mfx.GopPicSize);
                msdk_printf(MSDK_STRING("Ref dist\t%d\n"), m_mfxEncParams.mfx.GopRefDist);
                msdk_printf(MSDK_STRING("Ref number\t%d\n"), m_mfxEncParams.mfx.NumRefFrame);
                msdk_printf(MSDK_STRING("Idr Interval\t%d\n"), m_mfxEncParams.mfx.IdrInterval);
                msdk_printf(MSDK_STRING("Target usage\t%s\n"), TargetUsageToStr(m_mfxEncParams.mfx.TargetUsage));

                const msdk_char* sMemType = m_memType == D3D9_MEMORY ? MSDK_STRING("d3d")
                    : (m_memType == D3D11_MEMORY ? MSDK_STRING("d3d11")
                    : MSDK_STRING("system"));
                msdk_printf(MSDK_STRING("Memory type\t%s\n"), sMemType);

                mfxIMPL impl;
                GetFirstSession().QueryIMPL(&impl);

                const msdk_char* sImpl = (MFX_IMPL_VIA_D3D11 == MFX_IMPL_VIA_MASK(impl)) ? MSDK_STRING("hw_d3d11")
                    : (MFX_IMPL_HARDWARE  & impl) ? MSDK_STRING("hw1")
                    : (MFX_IMPL_HARDWARE2 & impl) ? MSDK_STRING("hw2")
                    : (MFX_IMPL_HARDWARE3 & impl) ? MSDK_STRING("hw3")
                    : (MFX_IMPL_HARDWARE4 & impl) ? MSDK_STRING("hw4")
                    : MSDK_STRING("sw");
                msdk_printf(MSDK_STRING("Media SDK impl\t\t%s\n"), sImpl);

                mfxVersion ver;
                GetFirstSession().QueryVersion(&ver);
                msdk_printf(MSDK_STRING("Media SDK version\t%d.%d\n"), ver.Major, ver.Minor);

                msdk_printf(MSDK_STRING("\n"));
            }
        }
    }
}