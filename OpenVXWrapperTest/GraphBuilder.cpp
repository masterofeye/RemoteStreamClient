#include "GraphBuilder.h"
#include "OpenVXWrapper.h"

namespace RW
{
    namespace CORE
    {
        GraphBuilder::GraphBuilder(QList<AbstractModule const*> *ModuleList, std::shared_ptr<spdlog::logger> Logger) : 
            m_ModuleList(ModuleList),
            m_Logger(Logger)
        {

        }


        GraphBuilder::~GraphBuilder()
        {
        }

        //tenStatus GraphBuilder::BuildNode(KernelManager *CurrentKernelManager, 
        //    tstInitialiseControlStruct* InitialiseControlStruct,
        //    tstControlStruct *ControlStruct,
        //    tstDeinitialiseControlStruct *DeinitialiseControlStruct
        //    )
        //{



        //    RW::CORE::Kernel kernel(&context, "MyKernel", NVX_KERNEL_TEST, 4, list[0], m_Logger);
        //    if (kernel.IsInitialized())
        //    {
        //        status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 0);
        //        status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 1);
        //        status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 2);
        //        status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 3);

        //        status = kernelManager.FinalizeKernel(&kernel);

        //        RW::CORE::Node node(&graph, &kernel, file_logger);
        //        if (node.IsInitialized())
        //        {
        //            //Test data
        //            RW::VG::tstMyInitialiseControlStruct initControll;
        //            RW::VG::tstMyControlStruct control;
        //            RW::VG::tstMyDeinitialiseControlStruct deinitControl;
        //            initControll.u8Test = 10;
        //            control.u8Test = 10;
        //            deinitControl.u8Test = 10;

        //            node.SetParameterByIndex(0, (void*)&kernel, sizeof(RW::CORE::Kernel), &context);
        //            node.SetParameterByIndex(1, (void*)&initControll, sizeof(RW::VG::stMyInitialiseControlStruct), &context);
        //            node.SetParameterByIndex(2, (void*)&control, sizeof(RW::VG::stMyControlStruct), &context);
        //            node.SetParameterByIndex(3, (void*)&deinitControl, sizeof(RW::VG::stMyDeinitialiseControlStruct), &context);
        //        }
        //    }
        //}
    }

}


