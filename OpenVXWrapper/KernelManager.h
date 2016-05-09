#pragma once

extern "C"{
#include "rwvx.h"
}
#include "Utils.h"

namespace RW
{
    namespace  CORE
    {
        class Kernel;
        class Context;

        class REMOTE_API KernelManager
        {
            vx_context      m_Context;

        public:
            KernelManager(Context *CurrentContext);
            ~KernelManager();

            tenStatus AddParameterToKernel(Kernel* KernelToAddParam, tenDir Direction, int Index);
            tenStatus LoadKernel(Kernel* const KernelToLoad);
            tenStatus FinalizeKernel(Kernel* const KernelToLoad);
            tenStatus SetKernelAttribute();
            tenStatus SetMetaFormatAttribute();
            tenStatus RemoveKernel(Kernel* const KernelToRemove);
        };

    }
}