#include "KernelManager.h"
#include "Kernel.h"
#include "Context.h"


namespace RW
{
    namespace CORE
    {
        KernelManager::KernelManager(Context *CurrentContext) 
            : m_Context((*CurrentContext)())
        {
        }


        KernelManager::~KernelManager()
        {
        }

        tenStatus  KernelManager::AddParameterToKernel(Kernel* const KernelToAddParam, tenDir Direction, int Index )
        {
            //TODO Type Paraemter muss spezifiert 
            vx_status status = vxAddParameterToKernel((*KernelToAddParam)(), Index, (int)Direction, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
            if (status != VX_SUCCESS)
            {
                //TODO error Log
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }

        /*
        
        */
        tenStatus KernelManager::LoadKernel(Kernel* const KernelToLoad)
        {
            vx_status status = vxLoadKernels(m_Context, KernelToLoad->KernelName().c_str());
            if (status != VX_SUCCESS)
            {
                //TODO error Log
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }

        /*
        @brief Finalize the registration of a UserKernel.
        @param KernToLoad 
        @return State of execution.
        @retval nenSuccess No Error durring execution.
        @retval nenError Any Error durring execution.
        */
        tenStatus KernelManager::FinalizeKernel(Kernel* const KernelToLoad)
        {
            vx_status status = vxFinalizeKernel((*KernelToLoad)());
            if (status != VX_SUCCESS)
            {
                //TODO error Log
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }
        tenStatus KernelManager::SetKernelAttribute()
        {
            //TODO Implementation
            return tenStatus::nenError;
        }

        tenStatus KernelManager::SetMetaFormatAttribute()
        {
            //TODO Implementation
            return tenStatus::nenError;
        }
        tenStatus KernelManager::RemoveKernel(Kernel* const KernelToRemove)
        {
            vx_status status = vxRemoveKernel((*KernelToRemove)());
            if (status != VX_SUCCESS)
            {
                //TODO error Log
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }
        
    }
}