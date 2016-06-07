#pragma once

extern "C"{
#include "rwvx.h"
}
#include "Utils.h"
#include "spdlog\spdlog.h"

namespace RW
{
    namespace  CORE
    {
        class Kernel;
        class Context;
		class Node;

        class REMOTE_API KernelManager
        {
            vx_context      m_Context;
            std::shared_ptr<spdlog::logger> m_Logger;
            std::vector<Kernel*> m_KernelList;
			std::vector<Node*> m_NodeList;
        public:
            KernelManager(Context *CurrentContext, std::shared_ptr<spdlog::logger> Logger);
            ~KernelManager();

            tenStatus AddParameterToKernel(Kernel* KernelToAddParam, tenDir Direction, int Index);
            tenStatus LoadKernel(Kernel* const KernelToLoad);
            tenStatus FinalizeKernel(Kernel* const KernelToLoad);
            tenStatus SetKernelAttribute();
            tenStatus SetMetaFormatAttribute();
            tenStatus RemoveKernel(Kernel* const KernelToRemove);
			inline void AddKernel(Kernel* CurrentKernel){ m_KernelList.push_back(CurrentKernel); }
			inline void AddNode(Node* CurrentNode){ m_NodeList.push_back(CurrentNode); }
        };

    }
}