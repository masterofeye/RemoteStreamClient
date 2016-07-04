#pragma once
#include "spdlog\spdlog.h"

typedef struct stPipelineParams
{
    std::shared_ptr<spdlog::logger> file_logger;

}tstPipelineParams;

class Pipeline
{
public:
    Pipeline();
    ~Pipeline();
    static int RunPipeline(tstPipelineParams params);
};

