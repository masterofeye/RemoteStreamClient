#include <iostream>
#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"
#include "Plugin1.hpp"

int main(int argc, const char* argv[])
{
    RW::tenStatus status = RW::tenStatus::nenError;

	RW::CORE::ModuleLoader ml;
	QList<RW::CORE::AbstractModule const*> list;
	ml.LoadPlugins(&list);


    
    RW::CORE::Context context;
    if (!context.IsInitialized())
        std::cout << "Context couldn't initialized." << std::endl;

    RW::CORE::Kernel kernel(&context, "MyKernel", NVX_KERNEL_TEST, list[0] );
    if (!kernel.IsInitialized())
        std::cout << "Kernel couldn't initialized." << std::endl;

    RW::CORE::KernelManager kernelManager(&context);


    kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput,0);
    kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput,1);

    status = kernelManager.FinalizeKernel(&kernel);
    if (status != RW::tenStatus::nenSuccess)
        std::cout << "Kernel couldn't load." << std::endl;

    RW::CORE::Graph graph(&context);
    if(!graph.IsInitialized())
        std::cout << "Graph couldn't initialized." << std::endl;



    RW::CORE::Node node(&graph, &kernel);
    if (!node.IsInitialized())
        std::cout << "Node couldn't initialized." << std::endl;
    //node.SetParameterByIndex(0,233,&context);
    
    RW::VG::tstMyInitialiseControlStruct test;
    test.u8Test = 10;
    node.SetParameterByIndex(0, (void*)&test, sizeof(RW::VG::stMyInitialiseControlStruct), &context);
    node.SetParameterByIndex(1, (void*)&kernel, sizeof(RW::CORE::Kernel), &context);

    if (graph.VerifyGraph() != RW::tenStatus::nenSuccess)
        std::cout << "Graph couldn't verfied." << std::endl;

    if (graph.ScheduleGraph() != RW::tenStatus::nenSuccess)
        std::cout << "Graph couldn't scheduled." << std::endl;
    if (graph.WaitGraph() != RW::tenStatus::nenSuccess)
        std::cout << "Graph couldn't wait." << std::endl;
    return 0;
}