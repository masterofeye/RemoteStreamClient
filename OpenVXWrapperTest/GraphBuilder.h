#pragma once
#include <QtCore>
#include "spdlog\spdlog.h"

namespace RW
{
    namespace CORE
    {
        class AbstractModule;

        class GraphBuilder
        {
        private:
            QList<AbstractModule const*> *m_ModuleList;
            std::shared_ptr<spdlog::logger> m_Logger;
        public:
            GraphBuilder(QList<AbstractModule const*> *ModuleList, std::shared_ptr<spdlog::logger> Logger);
            ~GraphBuilder();

        };
    }
}
