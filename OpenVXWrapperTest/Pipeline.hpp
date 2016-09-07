#pragma once

#include "spdlog\spdlog.h"

//#include <QApplication>
#include <QtWidgets/QApplication>

#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"

/*Modules*/
#include "GraphBuilder.h"

#include "live555\SCL_live555.hpp"
#include "INTEL\DEC_Intel.hpp"
#include "INTEL\DEC_inputs.h"
#include "NVENC\DEC_NvDecodeD3D9.hpp"
#include "NVENC\DEC_NVENC_inputs.h"
#include "ConvColor_NV12toRGB\IMP_ConvColorFramesNV12ToRGB.hpp"
#include "QT_simple\VPL_FrameProcessor.hpp"
#include "QT_simple\VPL_Viewer.hpp"

#include "HighResolution\HighResClock.h"
#include "dynlink_cuda.h"
#include "opencv2\cudev\common.hpp"

#define SAFE_DELETE(P) {if (P) {delete P; P = nullptr;}}
#define TRACE 1
#define TRACE_PERFORMANCE

typedef struct stPipelineParams
{
    std::shared_ptr<spdlog::logger> file_logger;
    RW::VPL::QT_SIMPLE::VPL_Viewer *pViewer;

}tstPipelineParams;

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
