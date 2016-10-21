/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "NvHWEncoder.h"

namespace RW{
    namespace ENC{
        namespace NVENC{
            NVENCSTATUS CNvHWEncoder::NvEncCreateBitstreamBuffer(uint32_t size, void** bitstreamBuffer)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams;

                memset(&createBitstreamBufferParams, 0, sizeof(createBitstreamBufferParams));
                SET_VER(createBitstreamBufferParams, NV_ENC_CREATE_BITSTREAM_BUFFER);

                createBitstreamBufferParams.size = size;
                createBitstreamBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

                nvStatus = m_pEncodeAPI->nvEncCreateBitstreamBuffer(m_hEncoder, &createBitstreamBufferParams);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::NvEncCreateBitstreamBuffer: m_pEncodeAPI->nvEncCreateBitstreamBuffer(...) did not succeed!");
                }

                *bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

                if (bitstreamBuffer)
                {
                    nvStatus = m_pEncodeAPI->nvEncDestroyBitstreamBuffer(m_hEncoder, bitstreamBuffer);
                    if (nvStatus != NV_ENC_SUCCESS)
                    {
                        m_Logger->error("CNvHWEncoder::NvEncDestroyBitstreamBuffer: m_pEncodeAPI->nvEncDestroyBitstreamBuffer(...) did not succeed!");
                    }
                }

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncRegisterAsyncEvent(void** completionEvent)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_EVENT_PARAMS eventParams;

                memset(&eventParams, 0, sizeof(eventParams));
                SET_VER(eventParams, NV_ENC_EVENT_PARAMS);

#if defined (NV_WINDOWS)
                eventParams.completionEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
#else
                eventParams.completionEvent = nullptr;
#endif
                nvStatus = m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &eventParams);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::NvEncRegisterAsyncEvent: m_pEncodeAPI->nvEncRegisterAsyncEvent(...) did not succeed!");
                }

                *completionEvent = eventParams.completionEvent;

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncUnregisterAsyncEvent(void* completionEvent)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_EVENT_PARAMS eventParams;

                if (completionEvent)
                {
                    memset(&eventParams, 0, sizeof(eventParams));
                    SET_VER(eventParams, NV_ENC_EVENT_PARAMS);

                    eventParams.completionEvent = completionEvent;

                    nvStatus = m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
                    if (nvStatus != NV_ENC_SUCCESS)
                    {
                        m_Logger->error("CNvHWEncoder::NvEncUnregisterAsyncEvent: m_pEncodeAPI->nvEncUnregisterAsyncEvent(...) did not succeed!");
                    }
                }

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncMapInputResource(void* registeredResource, void** mappedResource)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_MAP_INPUT_RESOURCE mapInputResParams;

                memset(&mapInputResParams, 0, sizeof(mapInputResParams));
                SET_VER(mapInputResParams, NV_ENC_MAP_INPUT_RESOURCE);

                mapInputResParams.registeredResource = registeredResource;

                nvStatus = m_pEncodeAPI->nvEncMapInputResource(m_hEncoder, &mapInputResParams);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::NvEncMapInputResource: m_pEncodeAPI->nvEncMapInputResource(...) did not succeed!");
                }

                *mappedResource = mapInputResParams.mappedResource;

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

                if (mappedInputBuffer)
                {
                    nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, mappedInputBuffer);
                    if (nvStatus != NV_ENC_SUCCESS)
                    {
                        m_Logger->error("CNvHWEncoder::NvEncUnmapInputResource: m_pEncodeAPI->nvEncUnmapInputResource(...) did not succeed!");
                    }
                }

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncDestroyEncoder()
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

                if (m_bEncoderInitialized)
                {
                    nvStatus = m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);

                    m_bEncoderInitialized = false;
                }

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncOpenEncodeSessionEx(void* device, NV_ENC_DEVICE_TYPE deviceType)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openSessionExParams;

                memset(&openSessionExParams, 0, sizeof(openSessionExParams));
                SET_VER(openSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);

                openSessionExParams.device = device;
                openSessionExParams.deviceType = deviceType;
                openSessionExParams.reserved = nullptr;
                openSessionExParams.apiVersion = NVENCAPI_VERSION;

                nvStatus = m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::NvEncOpenEncodeSessionEx: m_pEncodeAPI->nvEncOpenEncodeSessionEx(...) did not succeed!");
                }

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, void** registeredResource)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_REGISTER_RESOURCE registerResParams;

                memset(&registerResParams, 0, sizeof(registerResParams));
                SET_VER(registerResParams, NV_ENC_REGISTER_RESOURCE);

                registerResParams.resourceType = resourceType;
                registerResParams.resourceToRegister = resourceToRegister;
                registerResParams.width = width;
                registerResParams.height = height;
                registerResParams.pitch = pitch;
                registerResParams.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12_PL;

                nvStatus = m_pEncodeAPI->nvEncRegisterResource(m_hEncoder, &registerResParams);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::NvEncRegisterResource: m_pEncodeAPI->nvEncRegisterResource(...) did not succeed!");
                }

                *registeredResource = registerResParams.registeredResource;

                return nvStatus;
            }

            CNvHWEncoder::CNvHWEncoder(std::shared_ptr<spdlog::logger> Logger) : m_Logger(Logger)
            {
                m_bEncoderInitialized = false;
                m_pEncodeAPI = nullptr;
                m_hinstLib = nullptr;
                m_EncodeIdx = 0;
                m_uCurWidth = 0;
                m_uCurHeight = 0;
                m_uMaxWidth = 0;
                m_uMaxHeight = 0;
                m_hEncoder = new HANDLE();

                memset(&m_stCreateEncodeParams, 0, sizeof(m_stCreateEncodeParams));
                SET_VER(m_stCreateEncodeParams, NV_ENC_INITIALIZE_PARAMS);

                memset(&m_stEncodeConfig, 0, sizeof(m_stEncodeConfig));
                SET_VER(m_stEncodeConfig, NV_ENC_CONFIG);
            }

            CNvHWEncoder::~CNvHWEncoder()
            {
                // clean up encode API resources here
                if (m_pEncodeAPI)
                {
                    delete m_pEncodeAPI;
                    m_pEncodeAPI = nullptr;
                }

                if (m_hinstLib)
                {
#if defined (NV_WINDOWS)
                    FreeLibrary(m_hinstLib);
#else
                    dlclose(m_hinstLib);
#endif
                }
                m_hinstLib = nullptr;
            }

            NVENCSTATUS CNvHWEncoder::ValidateEncodeGUID(GUID inputCodecGuid)
            {
                unsigned int i, codecFound, encodeGUIDCount, encodeGUIDArraySize;
                NVENCSTATUS nvStatus;
                GUID *encodeGUIDArray;

                nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, &encodeGUIDCount);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::ValidateEncodeGUID: m_pEncodeAPI->nvEncGetEncodeGUIDCount(...) did not succeed!");
                    return nvStatus;
                }

                encodeGUIDArray = new GUID[encodeGUIDCount];
                memset(encodeGUIDArray, 0, sizeof(GUID)* encodeGUIDCount);

                encodeGUIDArraySize = 0;
                nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, encodeGUIDArray, encodeGUIDCount, &encodeGUIDArraySize);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    delete[] encodeGUIDArray;
                    m_Logger->error("CNvHWEncoder::ValidateEncodeGUID: m_pEncodeAPI->nvEncGetEncodeGUIDs(...) did not succeed!");
                    return nvStatus;
                }

                m_Logger->debug("CNvHWEncoder::ValidateEncodeGUID: encodeGUIDArraySize <= encodeGUIDCount ? ") \
                    << (encodeGUIDArraySize <= encodeGUIDCount);

                codecFound = 0;
                for (i = 0; i < encodeGUIDArraySize; i++)
                {
                    if (inputCodecGuid == encodeGUIDArray[i])
                    {
                        codecFound = 1;
                        break;
                    }
                }

                delete[] encodeGUIDArray;

                if (codecFound)
                {
                    return NV_ENC_SUCCESS;
                }
                else
                {
                    m_Logger->error("CNvHWEncoder::ValidateEncodeGUID: codec not found! Invalid GUID.");
                    return NV_ENC_ERR_INVALID_PARAM;
                }
            }

            NVENCSTATUS CNvHWEncoder::ValidatePresetGUID(GUID inputPresetGuid, GUID inputCodecGuid)
            {
                uint32_t i, presetFound, presetGUIDCount, presetGUIDArraySize;
                NVENCSTATUS nvStatus;
                GUID *presetGUIDArray;

                nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, inputCodecGuid, &presetGUIDCount);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::ValidatePresetGUID: m_pEncodeAPI->nvEncGetEncodePresetCount(...) did not succeed!");
                    return nvStatus;
                }

                presetGUIDArray = new GUID[presetGUIDCount];
                memset(presetGUIDArray, 0, sizeof(GUID)* presetGUIDCount);

                presetGUIDArraySize = 0;
                nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, inputCodecGuid, presetGUIDArray, presetGUIDCount, &presetGUIDArraySize);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    delete[] presetGUIDArray;
                    m_Logger->error("CNvHWEncoder::ValidatePresetGUID: m_pEncodeAPI->nvEncGetEncodePresetGUIDs(...) did not succeed!");
                    return nvStatus;
                }

                m_Logger->debug("CNvHWEncoder::ValidatePresetGUID: presetGUIDArraySize <= presetGUIDCount :") << (presetGUIDArraySize <= presetGUIDCount);

                presetFound = 0;
                for (i = 0; i < presetGUIDArraySize; i++)
                {
                    if (inputPresetGuid == presetGUIDArray[i])
                    {
                        presetFound = 1;
                        break;
                    }
                }

                delete[] presetGUIDArray;

                if (presetFound)
                {
                    return NV_ENC_SUCCESS;
                }
                else
                {
                    m_Logger->error("CNvHWEncoder::ValidatePresetGUID: Present GUID not found!");
                    return NV_ENC_ERR_INVALID_PARAM;
                }
            }

            NVENCSTATUS CNvHWEncoder::CreateEncoder(const EncodeConfig *pEncCfg)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

                if (pEncCfg == nullptr)
                {
                    m_Logger->error("CNvHWEncoder::CreateEncoder: pEncCfg is empty!");
                    return NV_ENC_ERR_INVALID_PARAM;
                }

                m_uCurWidth = pEncCfg->nWidth;
                m_uCurHeight = pEncCfg->nHeight;

                m_uMaxWidth = (pEncCfg->maxWidth > 0 ? pEncCfg->maxWidth : pEncCfg->nWidth);
                m_uMaxHeight = (pEncCfg->maxHeight > 0 ? pEncCfg->maxHeight : pEncCfg->nHeight);

                if ((m_uCurWidth > m_uMaxWidth) || (m_uCurHeight > m_uMaxHeight)) {
                    return NV_ENC_ERR_INVALID_PARAM;
                }
                if (!pEncCfg->nWidth || !pEncCfg->nHeight)
                {
                    m_Logger->error("CNvHWEncoder::CreateEncoder: parameters invalid (width or height)!");
                    return NV_ENC_ERR_INVALID_PARAM;
                }

                GUID inputCodecGUID = pEncCfg->codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
                nvStatus = ValidateEncodeGUID(inputCodecGUID);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::CreateEncoder: ValidateEncodeGUID did not succeed! Codec not supported.");
                    return nvStatus;
                }

                codecGUID = inputCodecGUID;

                m_stCreateEncodeParams.encodeGUID = inputCodecGUID;
                m_stCreateEncodeParams.presetGUID = pEncCfg->presetGUID;
                m_stCreateEncodeParams.encodeWidth = pEncCfg->nWidth;
                m_stCreateEncodeParams.encodeHeight = pEncCfg->nHeight;

                m_stCreateEncodeParams.darWidth = pEncCfg->nWidth;
                m_stCreateEncodeParams.darHeight = pEncCfg->nHeight;
                m_stCreateEncodeParams.frameRateNum = pEncCfg->fps;
                m_stCreateEncodeParams.frameRateDen = 1;
#if defined(NV_WINDOWS)
                m_stCreateEncodeParams.enableEncodeAsync = 1;
#else
                m_stCreateEncodeParams.enableEncodeAsync = 0;
#endif
                m_stCreateEncodeParams.enablePTD = 1;
                m_stCreateEncodeParams.reportSliceOffsets = 0;
                m_stCreateEncodeParams.enableSubFrameWrite = 0;
                m_stCreateEncodeParams.encodeConfig = &m_stEncodeConfig;
                m_stCreateEncodeParams.maxEncodeWidth = m_uMaxWidth;
                m_stCreateEncodeParams.maxEncodeHeight = m_uMaxHeight;

                // apply preset
                NV_ENC_PRESET_CONFIG stPresetCfg;
                memset(&stPresetCfg, 0, sizeof(NV_ENC_PRESET_CONFIG));
                SET_VER(stPresetCfg, NV_ENC_PRESET_CONFIG);
                SET_VER(stPresetCfg.presetCfg, NV_ENC_CONFIG);

                nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, m_stCreateEncodeParams.encodeGUID, m_stCreateEncodeParams.presetGUID, &stPresetCfg);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::CreateEncoder: m_pEncodeAPI->nvEncGetEncodePresetConfig(...) did not succeed!");
                    return nvStatus;
                }
                memcpy(&m_stEncodeConfig, &stPresetCfg.presetCfg, sizeof(NV_ENC_CONFIG));

                m_stEncodeConfig.gopLength = pEncCfg->gopLength;
                m_stEncodeConfig.frameIntervalP = pEncCfg->numB + 1;
                if (pEncCfg->pictureStruct == NV_ENC_PIC_STRUCT_FRAME)
                {
                    m_stEncodeConfig.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
                }
                else
                {
                    m_stEncodeConfig.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
                }

                m_stEncodeConfig.mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;

                if (pEncCfg->bitrate || pEncCfg->vbvMaxBitrate)
                {
                    m_stEncodeConfig.rcParams.rateControlMode = (NV_ENC_PARAMS_RC_MODE)pEncCfg->rcMode;
                    m_stEncodeConfig.rcParams.averageBitRate = pEncCfg->bitrate;
                    m_stEncodeConfig.rcParams.maxBitRate = pEncCfg->vbvMaxBitrate;
                    m_stEncodeConfig.rcParams.vbvBufferSize = pEncCfg->vbvSize;
                    m_stEncodeConfig.rcParams.vbvInitialDelay = pEncCfg->vbvSize * 9 / 10;
                }
                else
                {
                    m_stEncodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
                }

                if (pEncCfg->rcMode == 0)
                {
                    m_stEncodeConfig.rcParams.constQP.qpInterP = pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : pEncCfg->qp;
                    m_stEncodeConfig.rcParams.constQP.qpInterB = pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : pEncCfg->qp;
                    m_stEncodeConfig.rcParams.constQP.qpIntra = pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : pEncCfg->qp;
                }

                // set up initial QP value
                if (pEncCfg->rcMode == NV_ENC_PARAMS_RC_VBR || pEncCfg->rcMode == NV_ENC_PARAMS_RC_VBR_MINQP ||
                    pEncCfg->rcMode == NV_ENC_PARAMS_RC_2_PASS_VBR) {
                    m_stEncodeConfig.rcParams.enableInitialRCQP = 1;
                    m_stEncodeConfig.rcParams.initialRCQP.qpInterP = pEncCfg->qp;
                    if (pEncCfg->i_quant_factor != 0.0 && pEncCfg->b_quant_factor != 0.0) {
                        m_stEncodeConfig.rcParams.initialRCQP.qpIntra = (int)(pEncCfg->qp * FABS(pEncCfg->i_quant_factor) + pEncCfg->i_quant_offset);
                        m_stEncodeConfig.rcParams.initialRCQP.qpInterB = (int)(pEncCfg->qp * FABS(pEncCfg->b_quant_factor) + pEncCfg->b_quant_offset);
                    }
                    else {
                        m_stEncodeConfig.rcParams.initialRCQP.qpIntra = pEncCfg->qp;
                        m_stEncodeConfig.rcParams.initialRCQP.qpInterB = pEncCfg->qp;
                    }

                }

                m_stEncodeConfig.encodeCodecConfig.h264Config.chromaFormatIDC = 1;

                if (pEncCfg->intraRefreshEnableFlag)
                {
                    if (pEncCfg->codec == NV_ENC_HEVC)
                    {
                        m_stEncodeConfig.encodeCodecConfig.hevcConfig.enableIntraRefresh = 1;
                        m_stEncodeConfig.encodeCodecConfig.hevcConfig.intraRefreshPeriod = pEncCfg->intraRefreshPeriod;
                        m_stEncodeConfig.encodeCodecConfig.hevcConfig.intraRefreshCnt = pEncCfg->intraRefreshDuration;
                    }
                    else
                    {
                        m_stEncodeConfig.encodeCodecConfig.h264Config.enableIntraRefresh = 1;
                        m_stEncodeConfig.encodeCodecConfig.h264Config.intraRefreshPeriod = pEncCfg->intraRefreshPeriod;
                        m_stEncodeConfig.encodeCodecConfig.h264Config.intraRefreshCnt = pEncCfg->intraRefreshDuration;
                    }
                }

                if (pEncCfg->invalidateRefFramesEnableFlag)
                {
                    if (pEncCfg->codec == NV_ENC_HEVC)
                    {
                        m_stEncodeConfig.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = 16;
                    }
                    else
                    {
                        m_stEncodeConfig.encodeCodecConfig.h264Config.maxNumRefFrames = 16;
                    }
                }

                //if (pEncCfg->qpDeltaMapFile)
                //{
                //    m_stEncodeConfig.rcParams.enableExtQPDeltaMap = 1;
                //}
                if (pEncCfg->codec == NV_ENC_H264)
                {
                    m_stEncodeConfig.encodeCodecConfig.h264Config.idrPeriod = pEncCfg->gopLength;
                }
                else if (pEncCfg->codec == NV_ENC_HEVC)
                {
                    m_stEncodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = pEncCfg->gopLength;
                }

                if (pEncCfg->enableMEOnly == 1 || pEncCfg->enableMEOnly == 2)
                {
                    NV_ENC_CAPS_PARAM stCapsParam;
                    memset(&stCapsParam, 0, sizeof(NV_ENC_CAPS_PARAM));
                    SET_VER(stCapsParam, NV_ENC_CAPS_PARAM);
                    stCapsParam.capsToQuery = NV_ENC_CAPS_SUPPORT_MEONLY_MODE;
                    m_stCreateEncodeParams.enableMEOnlyMode = true;
                    int meonlyMode = 0;
                    nvStatus = m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, m_stCreateEncodeParams.encodeGUID, &stCapsParam, &meonlyMode);
                    if (nvStatus != NV_ENC_SUCCESS)
                    {
                        m_Logger->error("CNvHWEncoder::CreateEncoder: m_pEncodeAPI->nvEncGetEncodeCaps(...) did not succeed! Encode Session Initialization failed");
                        return nvStatus;
                    }
                    else
                    {
                        if (meonlyMode == 1)
                        {
                            m_Logger->debug("CNvHWEncoder::CreateEncoder: NV_ENC_CAPS_SUPPORT_MEONLY_MODE  supported\n");
                        }
                        else
                        {
                            m_Logger->error("CNvHWEncoder::CreateEncoder: NV_ENC_CAPS_SUPPORT_MEONLY_MODE not supported");
                            return NV_ENC_ERR_UNSUPPORTED_DEVICE;
                        }
                    }
                }

                nvStatus = m_pEncodeAPI->nvEncInitializeEncoder(m_hEncoder, &m_stCreateEncodeParams);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::CreateEncoder: m_pEncodeAPI->nvEncInitializeEncoder(...) did not succeed! Encode Session Initialization failed");
                    return nvStatus;
                }
                m_bEncoderInitialized = true;

                return nvStatus;
            }

            GUID CNvHWEncoder::GetPresetGUID(char* encoderPreset, int codec)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                GUID presetGUID = NV_ENC_PRESET_DEFAULT_GUID;

                if (encoderPreset && (stricmp(encoderPreset, "hq") == 0))
                {
                    presetGUID = NV_ENC_PRESET_HQ_GUID;
                }
                else if (encoderPreset && (stricmp(encoderPreset, "lowLatencyHP") == 0))
                {
                    presetGUID = NV_ENC_PRESET_LOW_LATENCY_HP_GUID;
                }
                else if (encoderPreset && (stricmp(encoderPreset, "hp") == 0))
                {
                    presetGUID = NV_ENC_PRESET_HP_GUID;
                }
                else if (encoderPreset && (stricmp(encoderPreset, "lowLatencyHQ") == 0))
                {
                    presetGUID = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
                }
                else if (encoderPreset && (stricmp(encoderPreset, "lossless") == 0))
                {
                    presetGUID = NV_ENC_PRESET_LOSSLESS_HP_GUID;
                }
                else
                {
                    if (encoderPreset)
                    {
                        m_Logger->error("CNvHWEncoder::GetPresetGUID: Unsupported preset guid ") << encoderPreset;
                    }
                    presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
                }

                GUID inputCodecGUID = codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
                nvStatus = ValidatePresetGUID(presetGUID, inputCodecGUID);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
                    m_Logger->error("CNvHWEncoder::GetPresetGUID: Unsupported preset guid ") << encoderPreset;
                }

                return presetGUID;
            }

            NVENCSTATUS CNvHWEncoder::ProcessOutput(const EncodeBuffer *pEncodeBuffer, RW::tstBitStream *pBitStream)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

                if (pEncodeBuffer->stOutputBfr.hBitstreamBuffer == nullptr && pEncodeBuffer->stOutputBfr.bEOSFlag == FALSE)
                {
                    m_Logger->error("CNvHWEncoder::ProcessOutput: Invalid Parameters!");
                    return NV_ENC_ERR_INVALID_PARAM;
                }

                if (pEncodeBuffer->stOutputBfr.bWaitOnEvent == TRUE)
                {
                    if (!pEncodeBuffer->stOutputBfr.hOutputEvent)
                    {
                        m_Logger->error("CNvHWEncoder::ProcessOutput: Invalid Parameters!");
                        return NV_ENC_ERR_INVALID_PARAM;
                    }
#if defined(NV_WINDOWS)
                    WaitForSingleObject(pEncodeBuffer->stOutputBfr.hOutputEvent, INFINITE);
#endif
                }

                if (pEncodeBuffer->stOutputBfr.bEOSFlag)
                    return NV_ENC_SUCCESS;

                nvStatus = NV_ENC_SUCCESS;
                NV_ENC_LOCK_BITSTREAM lockBitstreamData;
                memset(&lockBitstreamData, 0, sizeof(lockBitstreamData));
                SET_VER(lockBitstreamData, NV_ENC_LOCK_BITSTREAM);
                lockBitstreamData.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
                lockBitstreamData.doNotWait = false;

                nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, &lockBitstreamData);

                if (nvStatus == NV_ENC_SUCCESS)
                {
                    pBitStream->u32Size = lockBitstreamData.bitstreamSizeInBytes;
                    pBitStream->pBuffer = new uint8_t[pBitStream->u32Size];
                    memcpy(pBitStream->pBuffer, lockBitstreamData.bitstreamBufferPtr, pBitStream->u32Size);

                    nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, pEncodeBuffer->stOutputBfr.hBitstreamBuffer);

#if 1
                    FILE *pFile;
                    pFile = fopen("C:\\dummy\\bitstream_CNvHWEncoder.264", "wb");
                    size_t sSize = sizeof(lockBitstreamData.outputBitstream);
                    fwrite(lockBitstreamData.bitstreamBufferPtr, 1, lockBitstreamData.bitstreamSizeInBytes, pFile);
                    fclose(pFile);
#endif
                }
                else
                {
                    m_Logger->error("CNvHWEncoder::ProcessOutput: m_pEncodeAPI->nvEncLockBitstream(...) did not succeed! Lock bitstream function failed.");
                }

                return nvStatus;
            }

            NVENCSTATUS CNvHWEncoder::Initialize(void* device, NV_ENC_DEVICE_TYPE deviceType)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                MYPROC nvEncodeAPICreateInstance; // function pointer to create instance in nvEncodeAPI

#if defined(NV_WINDOWS)
#if defined (_WIN64)
                m_hinstLib = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
#else
                m_hinstLib = LoadLibrary(TEXT("nvEncodeAPI.dll"));
#endif
#else
                m_hinstLib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif
                if (m_hinstLib == nullptr)
                    return NV_ENC_ERR_OUT_OF_MEMORY;

#if defined(NV_WINDOWS)
                nvEncodeAPICreateInstance = (MYPROC)GetProcAddress(m_hinstLib, "NvEncodeAPICreateInstance");
#else
                nvEncodeAPICreateInstance = (MYPROC)dlsym(m_hinstLib, "NvEncodeAPICreateInstance");
#endif

                if (nvEncodeAPICreateInstance == nullptr)
                    return NV_ENC_ERR_OUT_OF_MEMORY;

                m_pEncodeAPI = new NV_ENCODE_API_FUNCTION_LIST;
                if (m_pEncodeAPI == nullptr)
                {
                    m_Logger->error("CNvHWEncoder::Initialize: m_pEncodeAPI is empty!");
                    return NV_ENC_ERR_OUT_OF_MEMORY;
                }
                memset(m_pEncodeAPI, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
                m_pEncodeAPI->version = NV_ENCODE_API_FUNCTION_LIST_VER;
                nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    return nvStatus;
                }
                nvStatus = NvEncOpenEncodeSessionEx(device, deviceType);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    return nvStatus;
                }
                return NV_ENC_SUCCESS;
            }

            NVENCSTATUS CNvHWEncoder::NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, NvEncPictureCommand *encPicCommand,
                uint32_t width, uint32_t height, tstBitStream *pPayload, NV_ENC_PIC_STRUCT ePicStruct,
                int8_t *qpDeltaMapArray, uint32_t qpDeltaMapArraySize)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_PIC_PARAMS encPicParams;

                memset(&encPicParams, 0, sizeof(encPicParams));
                SET_VER(encPicParams, NV_ENC_PIC_PARAMS);

                encPicParams.inputBuffer = pEncodeBuffer->stInputBfr.hInputSurface;
                encPicParams.bufferFmt = pEncodeBuffer->stInputBfr.bufferFmt;
                encPicParams.inputWidth = width;
                encPicParams.inputHeight = height;
                encPicParams.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
                encPicParams.completionEvent = pEncodeBuffer->stOutputBfr.hOutputEvent;
                encPicParams.inputTimeStamp = m_EncodeIdx;
                encPicParams.pictureStruct = ePicStruct;
                encPicParams.qpDeltaMap = qpDeltaMapArray;
                encPicParams.qpDeltaMapSize = qpDeltaMapArraySize;

                if (pPayload){
                    NV_ENC_SEI_PAYLOAD *pnvPayload = new NV_ENC_SEI_PAYLOAD();
                    pnvPayload->payload = (uint8_t*)pPayload->pBuffer;
                    pnvPayload->payloadSize = pPayload->u32Size;
                    pnvPayload->payloadType = 5;
                    encPicParams.codecPicParams.h264PicParams.seiPayloadArray = pnvPayload;
                }

                if (encPicCommand)
                {
                    if (encPicCommand->bForceIDR)
                    {
                        encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
                    }

                    if (encPicCommand->bForceIntraRefresh)
                    {
                        if (codecGUID == NV_ENC_CODEC_HEVC_GUID)
                        {
                            encPicParams.codecPicParams.hevcPicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
                        }
                        else
                        {
                            encPicParams.codecPicParams.h264PicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
                        }
                    }
                }

                nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
                if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT)
                {
                    m_Logger->error("CNvHWEncoder::NvEncEncodeFrame: m_pEncodeAPI->nvEncEncodePicture(...) did not succeed!");
                    return nvStatus;
                }

                m_EncodeIdx++;

                return NV_ENC_SUCCESS;
            }

            NVENCSTATUS CNvHWEncoder::NvEncFlushEncoderQueue(void *hEOSEvent)
            {
                NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
                NV_ENC_PIC_PARAMS encPicParams;
                memset(&encPicParams, 0, sizeof(encPicParams));
                SET_VER(encPicParams, NV_ENC_PIC_PARAMS);
                encPicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
                encPicParams.completionEvent = hEOSEvent;
                nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
                if (nvStatus != NV_ENC_SUCCESS)
                {
                    m_Logger->error("CNvHWEncoder::NvEncFlushEncoderQueue: m_pEncodeAPI->nvEncEncodePicture(...) did not succeed!");
                }
                return nvStatus;
            }
        }
    }
}
