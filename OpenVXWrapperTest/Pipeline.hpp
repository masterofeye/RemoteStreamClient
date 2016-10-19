#pragma once

#include "spdlog\spdlog.h"

//#include <QApplication>
#include <QtWidgets/QApplication>

#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"

/*Modules*/
#include "GraphBuilder.h"
#ifdef CLIENT
#include "..\CCL\Config.h"
#else if defined (SERVER)
#include "..\CSR\Config.h"
#endif

#ifdef SERVER
#include "Simu\VideoGrabberSimu.hpp"
#include "Crop\IMP_CropFrames.hpp"
#include "Merge\IMP_MergeFrames.hpp"
#include "ConvColor_BGRtoYUV420\IMP_ConvColorFramesBGRToYUV420.hpp"
#include "NVENC\ENC_CudaInterop.hpp"
#include "Intel\ENC_Intel.hpp"
#include "Intel\ENC_Intel_input.h"
#include "live555\SSR_live555.hpp"
#endif

#ifdef CLIENT
#include "live555\SCL_live555.hpp"
#include "INTEL\DEC_Intel.hpp"
#include "INTEL\DEC_inputs.h"
#include "NVENC\DEC_NvDecodeD3D9.hpp"
#include "NVENC\DEC_NVENC_inputs.h"
#include "ConvColor_NV12toRGB\IMP_ConvColorFramesNV12ToRGB.hpp"
#include "QT_simple\VPL_FrameProcessor.hpp"
#include "QT_simple\VPL_Viewer.hpp"
#endif

#include "HighResolution\HighResClock.h"
#include "dynlink_cuda.h"
#include "opencv2\cudev\common.hpp"

#define TRACE 1
#define TRACE_PERFORMANCE

typedef struct stPipelineParams
{
    std::shared_ptr<spdlog::logger> file_logger;
#ifdef CLIENT
    RW::VPL::QT_SIMPLE::VPL_Viewer *pViewer;
#endif
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
