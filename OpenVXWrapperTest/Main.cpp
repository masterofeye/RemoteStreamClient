#include <iostream>
#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"
/*Modules*/
#include "Plugin1.hpp"
#include "GraphBuilder.h"
#include "ENC_CudaInterop.hpp"
#include "VideoGrabberSimu.hpp"
#include "IMP_CropFrames.hpp"
#include "IMP_ConvColorFrames.hpp"
#include "DEC_Intel.hpp"
#include "DEC_inputs.h"
#include "VideoPlayer.hpp"

#include "HighResolution\HighResClock.h"
#include "spdlog\spdlog.h"
#include <thread>
#include "common\inc\dynlink_cuda.h"

#include <QApplication>
//#include "vld.h"

#define TRACE 1
#define TRACE_PERFORMANCE



void logtest()
{
    auto logger = spdlog::get("file_logger");
    logger->info("Log from application");
}

typedef struct stPipelineParams
{
    RW::VG::tstVideoGrabberInitialiseControlStruct videoGrabberInitialiseControlStruct;
    RW::VG::tstVideoGrabberControlStruct videoGrabberControlStruct;
    RW::VG::tstVideoGrabberControlStruct videoGrabberDeinitialiseControlStruct;

	RW::IMP::tstMyInitialiseControlStruct impCropInitialiseControlStruct;
	RW::IMP::tstMyControlStruct impCropControlStruct;
	RW::IMP::tstMyDeinitialiseControlStruct impCropDeinitialiseControlStruct;

	RW::IMP::tstMyInitialiseControlStruct impColorInitialiseControlStruct;
	RW::IMP::tstMyControlStruct impColorControlStruct;
	RW::IMP::tstMyDeinitialiseControlStruct impColorDeinitialiseControlStruct;

	RW::ENC::tstMyInitialiseControlStruct encodeInitialiseControlStruct;
    RW::ENC::tstMyControlStruct encodeControlStruct;
    RW::ENC::tstMyDeinitialiseControlStruct encodeDeinitialiseControlStruct;

    RW::DEC::tstMyInitialiseControlStruct decodeInitialiseControlStruct;
    RW::DEC::tstMyControlStruct decodeControlStruct;
    RW::DEC::tstMyDeinitialiseControlStruct decodeDeinitialiseControlStruct;

    RW::VPL::tstMyInitialiseControlStruct playerInitialiseControlStruct;
    RW::VPL::tstMyControlStruct playerControlStruct;
    RW::VPL::tstMyDeinitialiseControlStruct playerDeinitialiseControlStruct;
}tstPipelineParams;

typedef struct stPayloadMsg
{
    int     iTimestamp;
    int     iFrameNbr;
    uint8_t u8CANSignal1;
    uint8_t u8CANSignal2;
    uint8_t u8CANSignal3;
}tstPayloadMsg;

int pipeline(tstPipelineParams params)
{
    auto file_logger = spdlog::stdout_logger_mt("file_logger");
    //auto file_logger = spdlog::rotating_logger_mt("file_logger", (qApp->applicationDirPath() + "/logfile.log").toStdString(), 1048576 * 5, 3);
    file_logger->debug("******************");
    file_logger->debug("*Applicationstart*");
    file_logger->debug("******************");

	char *testLeak = new char[1000000];
	testLeak = nullptr;


    try
    {
        RW::tenStatus status = RW::tenStatus::nenError;
#ifdef TRACE_PERFORMANCE
        RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
        RW::CORE::ModuleLoader ml(file_logger);

        /*Load Plugins*/
        QList<RW::CORE::AbstractModule *> list;
        ml.LoadPlugins(&list);
#ifdef TRACE_PERFORMANCE
        RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
        file_logger->trace() << "Time to load Plugins: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
        t1 = RW::CORE::HighResClock::now();
#endif
        RW::CORE::Context context(file_logger);
        if (context.IsInitialized())
        {
            RW::CORE::Graph graph(&context, file_logger);
            RW::CORE::KernelManager kernelManager(&context, file_logger);

            RW::CORE::GraphBuilder builder(&list, file_logger, &graph, &context);

            RW::VG::tstVideoGrabberInitialiseControlStruct videoGrabberInitialiseControlStruct = params.videoGrabberInitialiseControlStruct;

            RW::VG::tstVideoGrabberControlStruct videoGrabberControlStruct;
            {
                videoGrabberControlStruct.pOutputData = new RW::tstBitStream();
                videoGrabberControlStruct.pOutputData->pBuffer = new uint8_t*();
                videoGrabberControlStruct.nCurrentFrameNumber = 0;
                videoGrabberControlStruct.nCurrentPositionMSec = 0;
            }
            RW::VG::tstVideoGrabberDeinitialiseControlStruct videoGrabberDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &videoGrabberInitialiseControlStruct,
                sizeof(videoGrabberInitialiseControlStruct),
                &videoGrabberControlStruct,
                sizeof(RW::VG::tstVideoGrabberControlStruct),
                &videoGrabberDeinitialiseControlStruct,
                sizeof(RW::VG::tstVideoGrabberDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenVideoGrabber_SIMU) != RW::tenStatus::nenSuccess)
                file_logger->error("nenVideoGrabber_SIMU couldn't build correct");

            RW::IMP::tstMyInitialiseControlStruct impCropInitialiseControlStruct;
            {
                impCropInitialiseControlStruct.pstFrameRect = new RW::IMP::tstRectStruct{ 500, 500, 100, 100 }; // only for nenGraphic_Crop
            }
            RW::IMP::tstMyControlStruct impCropControlStruct;
            {
                impCropControlStruct.pcInput = new RW::IMP::cInputBase(
                    videoGrabberInitialiseControlStruct.nFrameWidth,
                    videoGrabberInitialiseControlStruct.nFrameHeight,
                    videoGrabberControlStruct.pOutputData->pBuffer);
                impCropControlStruct.pcOutput = new RW::IMP::cOutputBase();
                impCropControlStruct.pcOutput->_pgMat = new cv::cuda::GpuMat();
            }
            RW::IMP::tstMyDeinitialiseControlStruct impCropDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
            	&impCropInitialiseControlStruct,
            	sizeof(RW::IMP::tstMyInitialiseControlStruct),
            	&impCropControlStruct,
            	sizeof(RW::IMP::tstMyControlStruct),
            	&impCropDeinitialiseControlStruct,
            	sizeof(RW::IMP::tstMyDeinitialiseControlStruct),
            	RW::CORE::tenSubModule::nenGraphic_Crop) != RW::tenStatus::nenSuccess)
            	file_logger->error("nenGraphic_Crop couldn't build correct");

            RW::IMP::tstMyInitialiseControlStruct impColorInitialiseControlStruct;
            RW::IMP::tstMyControlStruct impColorControlStruct;
            {
                impColorControlStruct.pcInput = new RW::IMP::cInputBase(
                    videoGrabberInitialiseControlStruct.nFrameWidth,
                    videoGrabberInitialiseControlStruct.nFrameHeight,
                    impCropControlStruct.pcOutput->_pgMat);
                impColorControlStruct.pcOutput = new RW::IMP::cOutputBase();
                impColorControlStruct.pcOutput->_pcuArray = new CUarray();
            }
            RW::IMP::tstMyDeinitialiseControlStruct impColorDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
            	&impColorInitialiseControlStruct,
            	sizeof(RW::IMP::tstMyInitialiseControlStruct),
            	&impColorControlStruct,
            	sizeof(RW::IMP::tstMyControlStruct),
            	&impColorDeinitialiseControlStruct,
            	sizeof(RW::IMP::tstMyDeinitialiseControlStruct),
            	RW::CORE::tenSubModule::nenGraphic_Color) != RW::tenStatus::nenSuccess)
            	file_logger->error("nenGraphic_Color couldn't build correct");

            RW::ENC::tstMyInitialiseControlStruct encodeInitialiseControlStruct;
            {
                encodeInitialiseControlStruct.pstEncodeConfig = new RW::ENC::EncodeConfig();
                encodeInitialiseControlStruct.pstEncodeConfig->width = videoGrabberInitialiseControlStruct.nFrameWidth;
                encodeInitialiseControlStruct.pstEncodeConfig->height = videoGrabberInitialiseControlStruct.nFrameHeight;
                encodeInitialiseControlStruct.pstEncodeConfig->fps = videoGrabberInitialiseControlStruct.nFPS;
                encodeInitialiseControlStruct.pstEncodeConfig->endFrameIdx = videoGrabberInitialiseControlStruct.nNumberOfFrames;
            }
            RW::ENC::tstMyControlStruct encodeControlStruct;
            {
                encodeControlStruct.pcuYUVArray = *impColorControlStruct.pcOutput->_pcuArray;
                encodeControlStruct.pPayload = new RW::tstBitStream();
                tstPayloadMsg Msg;
                Msg.iTimestamp = videoGrabberControlStruct.nCurrentPositionMSec;
                Msg.iFrameNbr = videoGrabberControlStruct.nCurrentFrameNumber;
                encodeControlStruct.pPayload->u32Size = sizeof(stPayloadMsg);
                encodeControlStruct.pPayload->pBuffer = (uint8_t*)&Msg;
            }
            RW::ENC::tstMyDeinitialiseControlStruct encodeDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                         &encodeInitialiseControlStruct,
                         sizeof(RW::ENC::tstMyInitialiseControlStruct),
                         &encodeControlStruct,
                         sizeof(RW::ENC::tstMyControlStruct),
                         &encodeDeinitialiseControlStruct,
                         sizeof(RW::ENC::tstMyDeinitialiseControlStruct),
                         RW::CORE::tenSubModule::nenEncode_NVIDIA) != RW::tenStatus::nenSuccess)
                         file_logger->error("nenEncode_NVIDIA couldn't build correct");

            //FILE *pFile;
            //pFile = fopen("C:\\Projekte\\BR213_24bbp_10_nv264_abr_1k.mp4", "r");
            //if (!pFile)
            //{
            //    file_logger->error("File did not load!");
            //    return -1;
            //}
            //// obtain file size:
            //fseek(pFile, 0, SEEK_END);
            //long lSize = ftell(pFile);
            //rewind(pFile);

            //// allocate memory to contain the whole file:
            //char *buffer = (char*)malloc(sizeof(char)*lSize);
            //// copy the file into the buffer:
            //size_t result = fread(buffer, 1, lSize, pFile);
            //if (!buffer)
            //{ 
            //    file_logger->error("Empty buffer!");
            //    return -1;
            //}
            //fclose(pFile);

            //RW::tstBitStream *pBitStream = new RW::tstBitStream();
            //pBitStream->pBuffer = buffer;
            //pBitStream->u32Size = lSize;

            //RW::tstBitStream *pPayload = new RW::tstBitStream();
            //pPayload->pBuffer = nullptr;
            //pPayload->u32Size = sizeof(stPayloadMsg);

            RW::DEC::tstMyInitialiseControlStruct decodeInitCtrl;
            { 
				decodeInitCtrl.inputParams = new RW::DEC::tstInputParams();
                decodeInitCtrl.inputParams->Height = /* 738; */
                    videoGrabberInitialiseControlStruct.nFrameHeight;
                decodeInitCtrl.inputParams->Width = /* 1920; */
                    videoGrabberInitialiseControlStruct.nFrameWidth;
                decodeInitCtrl.inputParams->nFrames = /* 2539; */
                    videoGrabberInitialiseControlStruct.nNumberOfFrames;
                decodeInitCtrl.inputParams->nMaxFPS = /* 60; */
                    videoGrabberInitialiseControlStruct.nFPS;
                decodeInitCtrl.inputParams->bUseHWLib = false;
            }
            RW::DEC::tstMyControlStruct decodeCtrl;
            {
                decodeCtrl.pstEncodedStream = /* pBitStream; */
                    encodeControlStruct.pstBitStream;
                decodeCtrl.pPayload = /* pPayload; */
                    encodeControlStruct.pPayload;
                decodeCtrl.pOutput = new RW::tstBitStream();
            }
            RW::DEC::tstMyDeinitialiseControlStruct decodeDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &decodeInitCtrl,
                sizeof(RW::DEC::tstMyInitialiseControlStruct),
                &decodeCtrl,
                sizeof(RW::DEC::tstMyControlStruct),
                &decodeDeinitCtrl,
                sizeof(RW::DEC::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenDecoder_INTEL) != RW::tenStatus::nenSuccess)
                file_logger->error("nenDecoder_INTEL couldn't build correct");

            RW::VPL::tstMyInitialiseControlStruct playerInitCtrl;
            RW::VPL::tstMyControlStruct playerCtrl;
            {
                playerCtrl.pstBitStream = decodeCtrl.pOutput;
                playerCtrl.TimeStamp = 10000;
                    //videoGrabberControlStruct.nCurrentPositionMSec;
            }
            RW::VPL::tstMyDeinitialiseControlStruct playerDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &playerInitCtrl,
                sizeof(RW::VPL::tstMyInitialiseControlStruct),
                &playerCtrl,
                sizeof(RW::VPL::tstMyControlStruct),
                &playerDeinitCtrl,
                sizeof(RW::VPL::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenPlayback_Simple) != RW::tenStatus::nenSuccess)
                file_logger->error("nenPlayback_Simple couldn't build correct");


            if (graph.VerifyGraph() == RW::tenStatus::nenSuccess)
            {
                if (graph.ScheduleGraph() == RW::tenStatus::nenSuccess)
                {

                    file_logger->debug("******************");
                    file_logger->debug("*Graph excecution*");
                    file_logger->debug("******************");
#ifdef TRACE_PERFORMANCE
                    t2 = RW::CORE::HighResClock::now();
                    file_logger->trace() << "Prepare Graph: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
                    t1 = RW::CORE::HighResClock::now();
#endif
                    graph.WaitGraph();
#ifdef TRACE_PERFORMANCE
                    t2 = RW::CORE::HighResClock::now();
                    file_logger->trace() << "Graph execution: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                }
            }

            //Cleanup. Whatever has been created here has to be destroyed here. Modules do not do that. 
            if (videoGrabberControlStruct.pOutputData)
            {
                if (videoGrabberControlStruct.pOutputData->pBuffer)
                {
                    delete videoGrabberControlStruct.pOutputData->pBuffer;
                    videoGrabberControlStruct.pOutputData->pBuffer = nullptr;
                }
                delete videoGrabberControlStruct.pOutputData;
                videoGrabberControlStruct.pOutputData = nullptr;
            }
            if (impCropInitialiseControlStruct.pstFrameRect)
			{
				delete impCropInitialiseControlStruct.pstFrameRect;
				impCropInitialiseControlStruct.pstFrameRect = nullptr;
			}
			if (impCropControlStruct.pcInput)
			{
				delete impCropControlStruct.pcInput;
				impCropControlStruct.pcInput = nullptr;
			}
			if (impCropControlStruct.pcOutput)
			{
				if (impCropControlStruct.pcOutput->_pgMat)
				{
					delete impColorControlStruct.pcOutput->_pgMat;
					impColorControlStruct.pcOutput->_pgMat = nullptr;
				}
				delete impCropControlStruct.pcOutput;
				impCropControlStruct.pcOutput = nullptr;
			}
			if (impColorControlStruct.pcInput)
			{
				delete impColorControlStruct.pcInput;
				impColorControlStruct.pcInput;
			}
			if (impColorControlStruct.pcOutput)
			{
				if (impColorControlStruct.pcOutput->_pcuArray)
				{
					delete impColorControlStruct.pcOutput->_pcuArray;
					impColorControlStruct.pcOutput->_pcuArray = nullptr;
				}
				delete impColorControlStruct.pcOutput;
				impColorControlStruct.pcOutput = nullptr;
            }
			if (encodeInitialiseControlStruct.pstEncodeConfig)
			{
				delete encodeInitialiseControlStruct.pstEncodeConfig;
				encodeInitialiseControlStruct.pstEncodeConfig = nullptr;
			}
			if (encodeControlStruct.pPayload)
			{
				delete encodeControlStruct.pPayload;
				encodeControlStruct.pPayload = nullptr;
			}

            // terminate testing data
            //free(buffer);

            //if (pPayload)
            //{
            //    delete pPayload;
            //    pPayload = nullptr;
            //}
            //if (pBitStream)
            //{
            //    delete pBitStream;
            //    pBitStream = nullptr;
            //}

			if (decodeInitCtrl.inputParams)
			{
                delete decodeInitCtrl.inputParams;
                decodeInitCtrl.inputParams = nullptr;
			}
            if (decodeCtrl.pOutput)
            {
                delete decodeCtrl.pOutput;
                decodeCtrl.pOutput = nullptr;
            }
        }
    }
    catch (...)
    {
        file_logger->flush();
    }
    return 0;
}

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

#ifdef DEBUG
        spdlog::set_level(spdlog::level::debug);
#elif TRACE
        spdlog::set_level(spdlog::level::trace);
#else
        spdlog::set_level(spdlog::level::info);
#endif

        RW::VG::tstVideoGrabberInitialiseControlStruct videoGrabberInitialiseControlStruct;
        {
            videoGrabberInitialiseControlStruct.nFPS = 30;
            videoGrabberInitialiseControlStruct.nFrameHeight = 1920;
            videoGrabberInitialiseControlStruct.nFrameWidth = 1080;
            videoGrabberInitialiseControlStruct.nNumberOfFrames = 1000;
            videoGrabberInitialiseControlStruct.sFileName = "E:\\Video\\BR213_24bbp_5.avi";
        }

        tstPipelineParams params;
        params.videoGrabberInitialiseControlStruct = videoGrabberInitialiseControlStruct;

        pipeline(params);
        //std::thread gv(pipeline, params);
        //std::thread wc(pipeline, params);

        //gv.join();
        //wc.join();

    return 0;
}