#include "KernelManager.h"
#include "Kernel.h"
#include "Context.h"


namespace RW
{
    namespace CORE
    {
        KernelManager::KernelManager(Context *CurrentContext, std::shared_ptr<spdlog::logger> Logger)
            : m_Context((*CurrentContext)()),
            m_Logger(Logger)
        {
        }

        KernelManager::~KernelManager()
        {
            m_Logger->debug("Destroy KernelManager");
        }

        tenStatus  KernelManager::AddParameterToKernel(Kernel* const KernelToAddParam, tenDir Direction, int Index )
        {
            //TODO Type Paraemter muss spezifiert 
            vx_status status = vxAddParameterToKernel((*KernelToAddParam)(), Index, (int)Direction, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED);
            if (status != VX_SUCCESS)
            {
                m_Logger->error("Couldn't add parameter to kernel... ") << "Index: " << Index;
                return tenStatus::nenError;
            }
            m_KernelList.push_back(KernelToAddParam);
            m_Logger->debug("Parameter added to Kernel (Index: ") << Index << ")";
            return tenStatus::nenSuccess;
        }

        /*
        
        */
        tenStatus KernelManager::LoadKernel(Kernel* const KernelToLoad)
        {
            vx_status status = vxLoadKernels(m_Context, KernelToLoad->KernelName().c_str());
            if (status != VX_SUCCESS)
            {
                m_Logger->error("Couldn't load kerlen");
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
                m_Logger->error("Couldn't finilize kernel");
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
                m_Logger->error("Couldn't remove kernel");
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }
        
    }
}