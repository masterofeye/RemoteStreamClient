#pragma once
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

#include <stdlib.h>
#include <stdio.h>
//#include <assert.h>

#include "dynlink_cuda.h" // <cuda.h>

#include "nvEncodeAPI.h"
#include "nvUtils.h"
#include "AbstractModule.hpp"

namespace RW{
    namespace ENC{

#define SET_VER(configStruct, type) {configStruct.version = type##_VER;}

#if defined (NV_WINDOWS)
#include "d3d9.h"
#define NVENCAPI __stdcall
#pragma warning(disable : 4996)
#elif defined (NV_UNIX)
#include <dlfcn.h>
#include <string.h>
#define NVENCAPI
#endif

#define DEFAULT_I_QFACTOR -0.8f
#define DEFAULT_B_QFACTOR 1.25f
#define DEFAULT_I_QOFFSET 0.f
#define DEFAULT_B_QOFFSET 1.25f
#define NV_ENC_CUDA 2

        enum
        {
            NV_ENC_H264 = 0,
            NV_ENC_HEVC = 1,
        };


        typedef struct _EncodeConfig
        {
            int              width;  //Width of frame
            int              height;  //Height of frame
            int              maxWidth;  //optional. maxWidth ? maxWidth : width
            int              maxHeight;  //optional. maxHeight ? maxHeight : height
            int              fps;  //Specify encoding frame rate in frames/sec
            int              bitrate;  //Specify the encoding average bitrate in bits/sec
            int              vbvMaxBitrate;  //Specify the vbv max bitrate in bits/sec
            int              vbvSize;  //Specify the encoding vbv/hrd buffer size in bits
            int              rcMode;  //valid parameters: NV_ENC_PARAMS_RC_CONSTQP or NV_ENC_PARAMS_RC_VBR or NV_ENC_PARAMS_RC_CBR or NV_ENC_PARAMS_RC_VBR_MINQP or NV_ENC_PARAMS_RC_2_PASS_QUALITY or NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP or NV_ENC_PARAMS_RC_2_PASS_VBR
            int              qp;  //Specify qp for Constant QP mode
            float            i_quant_factor;  //Specify qscale difference between I-frames and P-frames
            float            b_quant_factor;  //Specify qscale difference between P-frames and B-frames
            float            i_quant_offset;  //Specify qscale offset between I-frames and P-frames
            float            b_quant_offset;  //Specify qscale offset between P-frames and B-frames
            GUID             presetGUID;  //valid parameters: NV_ENC_PRESET_LOW_LATENCY_HQ_GUID or NV_ENC_PRESET_LOW_LATENCY_HP_GUID or NV_ENC_PRESET_HQ_GUID or NV_ENC_PRESET_HP_GUID or NV_ENC_PRESET_LOSSLESS_HP_GUID or NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID
            int              codec;  //NV_ENC_H264 or NV_ENC_HEVC
            int              invalidateRefFramesEnableFlag;
            int              intraRefreshEnableFlag;  //Specify if intra refresh is used during encoding. Intra Refresh and Invalidate Reference Frames cant be enabled simultaneously. 
            int              intraRefreshPeriod;  //Specify period for cyclic intra refresh
            int              intraRefreshDuration;  //Specify the intra refresh duration
            int              deviceType;   //valid parameters: NV_ENC_DX9 or NV_ENC_CUDA or NV_ENC_DX11 or NV_ENC_DX10
            int              startFrameIdx;
            int              endFrameIdx;
            int              gopLength;  //Group of Pictures length. Specifies the number of pictures in one GOP. Low latency application client can set goplength to NVENC_INFINITE_GOPLENGTH so that keyframes are not inserted automatically.
            int              numB;   //Specify number of B frame
            int              pictureStruct;  //valid parameters: NV_ENC_PIC_STRUCT_FRAME or NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM or NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP
            int              deviceID;    // consecutive number, has to be <= deviceCount
            int              isYuv444;
            //char            *qpDeltaMapFile;  // Specify the file containing the external QP delta map
            //char* inputFileName;	// PTX file. Fixed path in ./data/
            char* encoderPreset;	// possible values: "hq" "lowLatencyHP" "lowLatencyHQ" "lossless"
            int  enableMEOnly;  //valid parameters: 0 or 1 or 2
            //int  preloadedFrameCount; // not being used; has to be > 1; if preloadedFrameCount == 1 use enableMEOnly

            _EncodeConfig() : width(0), height(0), maxWidth(0), maxHeight(0), 
                fps(30), bitrate(5000000), vbvMaxBitrate(0), vbvSize(0), 
                rcMode(NV_ENC_PARAMS_RC_CONSTQP), qp(28), 
                i_quant_factor(DEFAULT_I_QFACTOR), b_quant_factor(DEFAULT_B_QFACTOR), i_quant_offset(DEFAULT_I_QOFFSET), b_quant_offset(DEFAULT_B_QOFFSET), 
                presetGUID(NV_ENC_PRESET_DEFAULT_GUID), 
                codec(NV_ENC_H264), 
                invalidateRefFramesEnableFlag(0), 
                intraRefreshEnableFlag(0), intraRefreshPeriod(0), intraRefreshDuration(0), 
                deviceType(NV_ENC_CUDA), 
                startFrameIdx(0), endFrameIdx(INT_MAX), 
                gopLength(NVENC_INFINITE_GOPLENGTH), 
                numB(0), 
                pictureStruct(NV_ENC_PIC_STRUCT_FRAME), deviceID(0), 
                isYuv444(0), encoderPreset("lowLatencyHP"), enableMEOnly(0) 
            {}

            ~_EncodeConfig()
            {
                if (encoderPreset)
                {
                    delete encoderPreset;
                    encoderPreset = nullptr;
                }
            }
        }EncodeConfig;

        typedef struct _EncodeInputBuffer
        {
            unsigned int      dwWidth;
            unsigned int      dwHeight;
#if defined (NV_WINDOWS)
            IDirect3DSurface9 *pNV12Surface;
#endif
            CUdeviceptr       pNV12devPtr;
            uint32_t          uNV12Stride;
            CUdeviceptr       pNV12TempdevPtr;
            uint32_t          uNV12TempStride;
            void*             nvRegisteredResource;
            NV_ENC_INPUT_PTR  hInputSurface;
            NV_ENC_BUFFER_FORMAT bufferFmt;
        }EncodeInputBuffer;

        typedef struct _EncodeOutputBuffer
        {
            unsigned int          dwBitstreamBufferSize;
            NV_ENC_OUTPUT_PTR     hBitstreamBuffer;
            HANDLE                hOutputEvent;
            bool                  bWaitOnEvent;
            bool                  bEOSFlag;
        }EncodeOutputBuffer;

        typedef struct _EncodeBuffer
        {
            EncodeOutputBuffer      stOutputBfr;
            EncodeInputBuffer       stInputBfr;
        }EncodeBuffer;

        typedef struct _NvEncPictureCommand
        {
            bool bResolutionChangePending;
            bool bBitrateChangePending;
            bool bForceIDR;
            bool bForceIntraRefresh;
            bool bInvalidateRefFrames;

            uint32_t newWidth;
            uint32_t newHeight;

            uint32_t newBitrate;
            uint32_t newVBVSize;

            uint32_t  intraRefreshDuration;

            uint32_t  numRefFramesToInvalidate;
            uint32_t  refFrameNumbers[16];
        }NvEncPictureCommand;

        //struct MEOnlyConfig
        //{
        //    unsigned char *yuv[2][3];
        //    unsigned int stride[3];
        //    unsigned int width;
        //    unsigned int height;
        //    unsigned int inputFrameIndex;
        //    unsigned int referenceFrameIndex;
        //};

        class CNvHWEncoder
        {
        public:
            uint32_t                                             m_EncodeIdx;
            //FILE                                                *m_fOutput;
            uint32_t                                             m_uMaxWidth;
            uint32_t                                             m_uMaxHeight;
            uint32_t                                             m_uCurWidth;
            uint32_t                                             m_uCurHeight;

        protected:
            bool                                                 m_bEncoderInitialized;
            GUID                                                 codecGUID;

            NV_ENCODE_API_FUNCTION_LIST                         *m_pEncodeAPI;
            HINSTANCE                                            m_hinstLib;
            void                                                *m_hEncoder;
            NV_ENC_INITIALIZE_PARAMS                             m_stCreateEncodeParams;
            NV_ENC_CONFIG                                        m_stEncodeConfig;
            std::shared_ptr<spdlog::logger>                      m_Logger;

        public:
            NVENCSTATUS NvEncOpenEncodeSession(void* device, uint32_t deviceType);
            NVENCSTATUS NvEncGetEncodeGUIDCount(uint32_t* encodeGUIDCount);
            NVENCSTATUS NvEncGetEncodeGUIDs(GUID* GUIDs, uint32_t guidArraySize, uint32_t* GUIDCount);
            NVENCSTATUS NvEncGetEncodeCaps(GUID encodeGUID, NV_ENC_CAPS_PARAM* capsParam, int* capsVal);
            NVENCSTATUS NvEncGetEncodePresetCount(GUID encodeGUID, uint32_t* encodePresetGUIDCount);
            NVENCSTATUS NvEncGetEncodePresetGUIDs(GUID encodeGUID, GUID* presetGUIDs, uint32_t guidArraySize, uint32_t* encodePresetGUIDCount);
            NVENCSTATUS NvEncGetEncodePresetConfig(GUID encodeGUID, GUID  presetGUID, NV_ENC_PRESET_CONFIG* presetConfig);
            NVENCSTATUS NvEncCreateBitstreamBuffer(uint32_t size, void** bitstreamBuffer);
            NVENCSTATUS NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer);
            NVENCSTATUS NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM* lockBitstreamBufferParams);
            NVENCSTATUS NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer);
            NVENCSTATUS NvEncRegisterAsyncEvent(void** completionEvent);
            NVENCSTATUS NvEncUnregisterAsyncEvent(void* completionEvent);
            NVENCSTATUS NvEncMapInputResource(void* registeredResource, void** mappedResource);
            NVENCSTATUS NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer);
            NVENCSTATUS NvEncDestroyEncoder();
            NVENCSTATUS NvEncOpenEncodeSessionEx(void* device, NV_ENC_DEVICE_TYPE deviceType);
            NVENCSTATUS NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width, uint32_t height, uint32_t pitch, void** registeredResource);
            NVENCSTATUS NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes);
            NVENCSTATUS NvEncReconfigureEncoder(const NvEncPictureCommand *pEncPicCommand);
            NVENCSTATUS NvEncFlushEncoderQueue(void *hEOSEvent);

            CNvHWEncoder(std::shared_ptr<spdlog::logger> m_Logger);
            virtual ~CNvHWEncoder();
            NVENCSTATUS                                          Initialize(void* device, NV_ENC_DEVICE_TYPE deviceType);
            NVENCSTATUS                                          Deinitialize();
            NVENCSTATUS                                          NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer, NvEncPictureCommand *encPicCommand,
                uint32_t width, uint32_t height, tstBitStream *pPayload,
                NV_ENC_PIC_STRUCT ePicStruct = NV_ENC_PIC_STRUCT_FRAME,
                int8_t *qpDeltaMapArray = nullptr, uint32_t qpDeltaMapArraySize = 0);
            NVENCSTATUS                                          CreateEncoder(const EncodeConfig *pEncCfg);
            GUID                                                 GetPresetGUID(char* encoderPreset, int codec);
            NVENCSTATUS                                          ProcessOutput(const EncodeBuffer *pEncodeBuffer, NV_ENC_LOCK_BITSTREAM *pstBitstreamData);
            NVENCSTATUS                                          FlushEncoder();
            NVENCSTATUS                                          ValidateEncodeGUID(GUID inputCodecGuid);
            NVENCSTATUS                                          ValidatePresetGUID(GUID presetCodecGuid, GUID inputCodecGuid);
        };

        typedef NVENCSTATUS(NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST*);
    }
}