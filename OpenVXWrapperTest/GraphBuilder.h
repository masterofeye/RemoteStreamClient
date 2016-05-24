#pragma once
#include <QtCore>
#include "OpenVXWrapper.h"
#include "spdlog\spdlog.h"
#include "AbstractModule.hpp"

namespace RW
{
    namespace CORE
    {
        class AbstractModule;

        class GraphBuilder
        {
        private:
            QList<RW::CORE::AbstractModule *> *m_ModuleList;
            Graph *m_Graph;
            Context *m_Context;
            std::shared_ptr<spdlog::logger> m_Logger;
        public:
            GraphBuilder(QList<AbstractModule *> *ModuleList, std::shared_ptr<spdlog::logger> Logger, Graph* CurrentGraph, Context* CurrentContext);
            ~GraphBuilder();
            tenStatus BuildNode(KernelManager *CurrentKernelManager,
                tstInitialiseControlStruct* InitialiseControlStruct,
                size_t size1,
                tstControlStruct *ControlStruct,
                size_t size2,
                tstDeinitialiseControlStruct *DeinitialiseControlStruct,
                size_t size3,
                tenSubModule SubModule,
                RW::CORE::Kernel **Kernel);
        };
    }
}
