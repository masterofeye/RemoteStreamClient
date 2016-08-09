#pragma once

#include "spdlog\spdlog.h"

//#include <QApplication>
#include <QtWidgets/QApplication>

#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"

/*Modules*/
#include "GraphBuilder.h"
#include "ENC_CudaInterop.hpp"
#include "VideoGrabberSimu.hpp"
#include "IMP_CropFrames.hpp"
#include "IMP_MergeFrames.hpp"
#include "IMP_ConvColorFrames.hpp"
//#include "DEC_Intel.hpp"
//#include "DEC_inputs.h"
#include "DEC_NvDecodeD3D9.hpp"
#include "DEC_NVENC_inputs.h"
#include "VPL_FrameProcessor.hpp"
#include "VPL_Viewer.hpp"

#include "HighResolution\HighResClock.h"
#include "..\Common_NVENC\inc\dynlink_cuda.h"
#include "opencv2\cudev\common.hpp"

#define SAFE_DELETE(P) {if (P) {delete P; P = nullptr;}}
#define TRACE 1
#define TRACE_PERFORMANCE

typedef struct stPipelineParams
{
    std::shared_ptr<spdlog::logger> file_logger;
    RW::VPL::VPL_Viewer *pViewer;
    //VideoPlayer *pViewer;

}tstPipelineParams;

typedef struct stPayloadMsg
{
    uint32_t u32Timestamp;
    uint32_t u32FrameNbr;
    uint8_t  u8CANSignal1;
    uint8_t  u8CANSignal2;
    uint8_t  u8CANSignal3;
}tstPayloadMsg;


class CPipeline : public QObject
{
    Q_OBJECT
public:
    CPipeline(tstPipelineParams *params);
    ~CPipeline(){};

public slots:
    int RunPipeline();

private:
    tstPipelineParams *m_params;
};

class CPipethread : public QThread
{
    Q_OBJECT
public:
    CPipethread(){};
    ~CPipethread(){};
    void start();

signals:
    int started();

};
