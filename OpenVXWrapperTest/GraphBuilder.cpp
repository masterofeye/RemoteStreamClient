#include "GraphBuilder.h"
#include "OpenVXWrapper.h"
#include "AbstractModule.hpp"

namespace RW
{
    namespace CORE
    {
        GraphBuilder::GraphBuilder(QList<AbstractModule*> *ModuleList, std::shared_ptr<spdlog::logger> Logger, Graph* CurrentGraph, Context* CurrentContext) : 
            m_ModuleList(ModuleList),
            m_Logger(Logger),
            m_Graph(CurrentGraph),
            m_Context(CurrentContext)
        {

        }


        GraphBuilder::~GraphBuilder()
        {
        }

        tenStatus GraphBuilder::BuildNode(KernelManager *CurrentKernelManager,
            tstInitialiseControlStruct* InitialiseControlStruct,
            size_t size1,
            tstControlStruct *ControlStruct,
            size_t size2,
            tstDeinitialiseControlStruct *DeinitialiseControlStruct,
            size_t size3,
            tenSubModule SubModule,
            RW::CORE::Kernel **Kernel
            )
        {
            tenStatus status = tenStatus::nenError;
            AbstractModule *module = nullptr;

            //Iteration trouhg all modules in the modulelist to find the current search module

            QListIterator<RW::CORE::AbstractModule*> i(*m_ModuleList);
            while (i.hasNext())
            {
                AbstractModule* currentModule = i.next();
                if (currentModule == nullptr)
                    break;
                if (currentModule->SubModulType() == SubModule)
                {
                    module = currentModule;
                    break;
                }
            }
            if (module == nullptr)
            {
                m_Logger->error() << "Module is missing" << Util::ModuleName(SubModule);
                return tenStatus::nenError;
            }
            m_Logger->debug("*************************");
            m_Logger->debug(Util::ModuleName(SubModule));
            m_Logger->debug("************************");

            *Kernel = new RW::CORE::Kernel(m_Context, "MyKernel", NVX_KERNEL_TEST, 4, module, m_Logger);
            if ((*Kernel)->IsInitialized())
            {
                status = CurrentKernelManager->AddParameterToKernel((*Kernel), RW::CORE::tenDir::nenInput, 0);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(1).");
                    return status;
                }

                status = CurrentKernelManager->AddParameterToKernel((*Kernel), RW::CORE::tenDir::nenInput, 1);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(2).");
                    return status;
                }

                status = CurrentKernelManager->AddParameterToKernel((*Kernel), RW::CORE::tenDir::nenInput, 2);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(3).");
                    return status;
                }

                status = CurrentKernelManager->AddParameterToKernel((*Kernel), RW::CORE::tenDir::nenInput, 3);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(4).");
                    return status;
                }

                status = CurrentKernelManager->FinalizeKernel((*Kernel));
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Finalize of the Kernel failed.");
                    return status;
                }

                (*Kernel)->SetParameter(1, InitialiseControlStruct);
                (*Kernel)->SetParameter(2, ControlStruct);
                (*Kernel)->SetParameter(3, DeinitialiseControlStruct);



                RW::CORE::Node node(m_Graph, (*Kernel), m_Logger);
                if (node.IsInitialized())
                {

                    status = node.SetParameterByIndex(0, (void*)(*Kernel), sizeof(RW::CORE::Kernel), m_Context);
                    if (status == tenStatus::nenError)
                    {
                        m_Logger->error("Set parameter for node failed(1).");
                        return status;
                    }
                    status = node.SetParameterByIndex(1, (void*)InitialiseControlStruct, size1, m_Context);
                    if (status == tenStatus::nenError)
                    {
                        m_Logger->error("Set parameter for node failed(2).");
                        return status;
                    }

                    status = node.SetParameterByIndex(2, (void*)ControlStruct, size2, m_Context);
                    if (status == tenStatus::nenError)
                    {
                        m_Logger->error("Set parameter for node failed(3).");
                        return status;
                    }

                    status = node.SetParameterByIndex(3, (void*)DeinitialiseControlStruct, size3, m_Context);
                    if (status == tenStatus::nenError)
                    {
                        m_Logger->error("Set parameter for node failed(4).");
                        return status;
                    }

                }
            }
            m_Logger->debug("Node successful created.");
            return tenStatus::nenSuccess;
        }
    }

}


