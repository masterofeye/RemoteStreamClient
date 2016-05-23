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
            tstControlStruct *ControlStruct,
            tstDeinitialiseControlStruct *DeinitialiseControlStruct,
            tenSubModule SubModule
            )
        {
            tenStatus status = tenStatus::nenError;
            AbstractModule *module = nullptr;

            //Iteration trouhg all modules in the modulelist to find the current search module

            QListIterator<RW::CORE::AbstractModule*> i(*m_ModuleList);
            while (i.hasNext())
            {
                AbstractModule* currentModule = i.next();
                if (currentModule->SubModulType() == SubModule)
                {
                    module = currentModule;
                    break;
                }
            }


            RW::CORE::Kernel kernel(m_Context, "MyKernel", NVX_KERNEL_TEST, 4, module, m_Logger);
            if (kernel.IsInitialized())
            {
                status = CurrentKernelManager->AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 0);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(1).");
                    return status;
                }

                status = CurrentKernelManager->AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 1);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(2).");
                    return status;
                }

                status = CurrentKernelManager->AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 2);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(3).");
                    return status;
                }

                status = CurrentKernelManager->AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 3);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Add parameter to Kernel failed(4).");
                    return status;
                }

                status = CurrentKernelManager->FinalizeKernel(&kernel);
                if (status == tenStatus::nenError)
                {
                    m_Logger->error("Finalize of the Kernel failed.");
                    return status;
                }

                RW::CORE::Node node(m_Graph, &kernel, m_Logger);
                if (node.IsInitialized())
                {

                    status = node.SetParameterByIndex(0, (void*)&kernel, sizeof(RW::CORE::Kernel), m_Context);
                    if (status == tenStatus::nenError)
                    {
                        m_Logger->error("Set parameter for node failed(1).");
                        return status;
                    }

                    status = node.SetParameterByIndex(1, (void*)&InitialiseControlStruct, sizeof(InitialiseControlStruct), m_Context);
                    if (status == tenStatus::nenError)
                    {
                        m_Logger->error("Set parameter for node failed(2).");
                        return status;
                    }

                    status = node.SetParameterByIndex(2, (void*)&ControlStruct, sizeof(ControlStruct), m_Context);
                    if (status == tenStatus::nenError)
                    {
                        m_Logger->error("Set parameter for node failed(3).");
                        return status;
                    }

                    status = node.SetParameterByIndex(3, (void*)&DeinitialiseControlStruct, sizeof(DeinitialiseControlStruct), m_Context);
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


