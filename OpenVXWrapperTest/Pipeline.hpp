#pragma once

#include "spdlog\spdlog.h"

//#include <QApplication>
#include <QtWidgets/QApplication>

#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"

/*Modules*/
#include "GraphBuilder.h"
#ifdef RS_SERVER
#include "..\CSR\Config.h"
#endif
#ifdef RS_CLIENT
#include "..\CCL\Config.h"
#endif

#ifdef RS_SERVER
#include "..\VGR\Simu\VideoGrabberSimu.hpp"
#include "..\IMP\Crop\IMP_CropFrames.hpp"
#include "..\IMP\Merge\IMP_MergeFrames.hpp"
#include "..\IMP\ConvColor_BGRtoNV12\IMP_ConvColorFramesBGRToNV12.hpp"
#include "..\ENC\NVENC\ENC_CudaInterop.hpp"
#include "..\ENC\Intel\ENC_Intel.hpp"
#include "..\ENC\Intel\ENC_Intel_input.h"
#include "..\SSR\live555\SSR_live555.hpp"
#endif

#ifdef RS_CLIENT
#include "..\SCL\live555\SCL_live555.hpp"
#include "..\DEC\INTEL\DEC_Intel.hpp"
#include "..\DEC\INTEL\DEC_inputs.h"
#include "..\DEC\NVENC\DEC_NvDecodeD3D9.hpp"
#include "..\DEC\NVENC\DEC_NVENC_inputs.h"
#include "..\IMP\ConvColor_NV12toRGB\IMP_ConvColorFramesNV12ToRGB.hpp"
#include "..\VPL\QT_simple\VPL_FrameProcessor.hpp"
#endif

#include "HighResolution\HighResClock.h"
#include "dynlink_cuda.h"
#include "opencv2\cudev\common.hpp"

#define TRACE 1
#define TRACE_PERFORMANCE

#ifdef REMOTE_EXPORT
#define REMOTE_API __declspec(dllexport)
#else
#define REMOTE_API __declspec(dllimport)
#endif

typedef struct stPipelineParams
{
    std::shared_ptr<spdlog::logger> file_logger;
#ifdef RS_CLIENT
    RW::VPL::QT_SIMPLE::VPL_Viewer *pViewer;
#endif
}tstPipelineParams;

class REMOTE_API CPipeline : public QObject
{
    Q_OBJECT
public:
    CPipeline(tstPipelineParams *params);
    ~CPipeline(){};

public slots:
    int RunPipeline();
    int StopPipeline();


private:
    bool m_bStopPipeline;
    tstPipelineParams *m_params;
};

class REMOTE_API CPipethread : public QThread
{
    Q_OBJECT
public:
    CPipethread(){};
    ~CPipethread(){};
    void start();
    void stop();

signals:
    int started();
    int stopped();

};
