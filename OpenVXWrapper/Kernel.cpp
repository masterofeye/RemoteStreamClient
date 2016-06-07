#include "Kernel.h"
#include "Context.h"
#include "Node.h"
#include "AbstractModule.hpp"

namespace RW
{
	namespace CORE
	{
		Kernel::Kernel(Context *CurrentContext, RW::CORE::tstControlStruct *ControlStruct, std::string KernelName, uint64_t KernelEnum, uint8_t ParameterAmount, AbstractModule const* Module, std::shared_ptr<spdlog::logger> Logger)
			: m_KernelName(KernelName),
			m_KernelEnum(KernelEnum),
			m_Initialize(false),
			m_Context((*CurrentContext)()),
			m_ControlStruct(ControlStruct),
            m_CurrentNode(nullptr),
			m_Logger(Logger)
		{

            m_AbstractModule = const_cast<AbstractModule*> (Module);
            AddKernel(ParameterAmount);
		}


		Kernel::~Kernel()
		{

		}

        void Kernel::SetParameter(int i, void* Value)
        {
            //TODO Sehr Unschön
            switch (i)
            {
            case 0:
                break;
            case 1:
                //m_InitialiseControlStruct = (tstInitialiseControlStruct*)Value;
                break;
            case 2:
                m_ControlStruct = (tstControlStruct*)Value;
                break;
            case 3:
                //m_DeinitialiseControlStruct = (tstDeinitialiseControlStruct*)Value;
                break;
            default:
                break;
            }
        }


        vx_status VX_CALLBACK Kernel::KernelInitializeCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
        {
            vx_status status = VX_FAILURE;
            vx_array kernenArray, controlStructArray;
            vx_parameter param[] = { vxGetParameterByIndex(Node, 0), vxGetParameterByIndex(Node, 1) };
            status = vxQueryParameter((vx_parameter)param[0], VX_PARAMETER_ATTRIBUTE_REF, &kernenArray, sizeof(kernenArray));
            if (status != VX_SUCCESS)
            {
                return VX_FAILURE;
            }
            vx_size size;

            Kernel *kernel = nullptr;
            vxAccessArrayRange(kernenArray, 0, 1, &size, (void**)&kernel, VX_READ_AND_WRITE);

            status = vxQueryParameter((vx_parameter)param[1], VX_PARAMETER_ATTRIBUTE_REF, &controlStructArray, sizeof(controlStructArray));
            RW::CORE::tstInitialiseControlStruct *controlStruct = nullptr;
            status = vxAccessArrayRange(controlStructArray, 0, 1, &size, (void**)&controlStruct, VX_READ_AND_WRITE);
            
            if (status != VX_SUCCESS)
            {
                //TODO log error and specific return value
                return VX_FAILURE;
            }
            

            //Kernel* kernel = reinterpret_cast<Kernel*>(data.ptr);
            try
            {
                if (kernel != nullptr)
                {
                    tenStatus ret = kernel->KernelInitialize((RW::CORE::tstInitialiseControlStruct*) controlStruct);
                    if (ret != tenStatus::nenSuccess)
                        return VX_FAILURE;
                }
                else
                {
                    return VX_FAILURE;
                }
            }
            catch (...)
            {
                //Todo Error log
                status = VX_FAILURE;
            }
            vxCommitArrayRange(kernenArray, 0, 1, kernel);
            vxCommitArrayRange(controlStructArray, 0, 1, controlStruct);

            return status;
        }

        RW::tenStatus Kernel::KernelInitialize(void* InitialiseControlStruct)
        {
            tenStatus status = tenStatus::nenError;
            status = m_AbstractModule->Initialise((tstInitialiseControlStruct*)InitialiseControlStruct);
            m_Logger->debug("Initialise kernel");
            return status;
        }

        vx_status  VX_CALLBACK Kernel::KernelDeinitializeCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
        {
            vx_status status = VX_FAILURE;
            vx_array kernenArray, controlStructArray;
            vx_parameter param[] = { vxGetParameterByIndex(Node, 0), vxGetParameterByIndex(Node, 3) };
            vx_size size = 0;

            /*Query the vx_array for the kernel parameter*/
            status = vxQueryParameter((vx_parameter)param[0], VX_PARAMETER_ATTRIBUTE_REF, &kernenArray, sizeof(kernenArray));
            if (status != VX_SUCCESS)
            {
                //TODO log error and specific return value
                return VX_FAILURE;
            }
            Kernel *kernel = nullptr;
            vxAccessArrayRange(kernenArray, 0, 1, &size, (void**)&kernel, VX_READ_AND_WRITE);

            /*Query the vx_array for the controlstruct parameter*/
            status = vxQueryParameter((vx_parameter)param[1], VX_PARAMETER_ATTRIBUTE_REF, &controlStructArray, sizeof(controlStructArray));
            if (status != VX_SUCCESS)
            {
                return VX_FAILURE;
            }
            RW::CORE::tstDeinitialiseControlStruct *controlStruct = nullptr;
            vxAccessArrayRange(controlStructArray, 0, 1, &size, (void**)&controlStruct, VX_READ_AND_WRITE);

            try
            {
                if (kernel != nullptr && controlStruct != nullptr)
                {
                    kernel->KernelDeinitialize((RW::CORE::tstDeinitialiseControlStruct*) controlStruct);
                }
                else
                {
                    return VX_FAILURE;
                }
            }
            catch (...)
            {
                //Todo Error log
            }
            vxCommitArrayRange(kernenArray, 0, 1, kernel);
            vxCommitArrayRange(controlStructArray, 0, 1, controlStruct);
            return status;
        }
        
        RW::tenStatus Kernel::KernelDeinitialize(void* DeinitializeControlStruct)
        {
            tenStatus status = tenStatus::nenError;
            m_Logger->debug("Deinitialize kernel" ) <<(int) m_AbstractModule->SubModulType();
            if (m_AbstractModule!= nullptr && m_AbstractModule->Deinitialise((tstDeinitialiseControlStruct*)DeinitializeControlStruct) != tenStatus::nenSuccess)
            {
                m_Logger->debug("Deinitialize kernel");
            }
            return status;
        }

        vx_status VX_CALLBACK Kernel::KernelInputValidateCB(vx_node Node, vx_uint32  Index)
        {
            vx_status status = VX_SUCCESS;

            return status;
        }

        vx_status Kernel::KernelOutputValidateCB(vx_node Node, vx_uint32  Index, vx_meta_format Meta)
        {
            vx_status status = VX_SUCCESS;

            return status;
        }

        vx_status VX_CALLBACK Kernel::KernelFncCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
        {
            vx_status status = VX_FAILURE;
            vx_array kernenArray, controlStructArray;
            vx_parameter param[] = { vxGetParameterByIndex(Node, 0), vxGetParameterByIndex(Node, 2) };
            status = vxQueryParameter((vx_parameter)param[0], VX_PARAMETER_ATTRIBUTE_REF, &kernenArray, sizeof(kernenArray));
            if (status != VX_SUCCESS)
            {
                return VX_FAILURE;
            }
            vx_size size;

            Kernel *kernel = nullptr;
            vxAccessArrayRange(kernenArray, 0, 1, &size, (void**)&kernel, VX_READ_AND_WRITE);

            
            status = vxQueryParameter((vx_parameter)param[1], VX_PARAMETER_ATTRIBUTE_REF, &controlStructArray, sizeof(controlStructArray));
            if (status != VX_SUCCESS)
            {
                return VX_FAILURE;
            }
            
            RW::CORE::tstControlStruct *controlStruct = nullptr;
            vxAccessArrayRange(controlStructArray, 0, 1, &size, (void**)&controlStruct, VX_READ_AND_WRITE);



            try
            {
                if (kernel != nullptr)
                {
                    kernel->KernelFnc((RW::CORE::tstControlStruct*)controlStruct);
                }
                else
                {
                    //TODO log error and specific return value
                    return VX_FAILURE;
                }
            }
            catch (...)
            {
                //Todo Error log
            }
            vxCommitArrayRange(kernenArray, 0, 1, kernel);
            vxCommitArrayRange(controlStructArray, 0, 1, controlStruct);


            return status;
        }

        tenStatus Kernel::KernelFnc(void* ControlStruct)
        {
            tenStatus status = tenStatus::nenError;
            status = m_AbstractModule->DoRender((tstControlStruct*)ControlStruct);
            m_Logger->debug("DoRender kernel");
            return status;
        }

        tenStatus Kernel::AddKernel(uint8_t ParamterAmount)
        {
            if (m_Context == nullptr)
            {
                //Todo Error
                return tenStatus::nenError;
            }

            vx_kernel kernel = vxAddKernel(m_Context, this->KernelName().c_str(), this->KernelEnum(), this->KernelFncCB, ParamterAmount, this->KernelInputValidateCB, this->KernelOutputValidateCB, this->KernelInitializeCB, this->KernelDeinitializeCB);

            vxGetKernelByEnum(m_Context, this->KernelEnum());

            vx_status status = vxGetStatus((vx_reference)kernel);
            if (status == VX_SUCCESS)
            {
                m_Initialize = true;
                m_Kernel = kernel;
                return tenStatus::nenSuccess;
            }
            else
            {
                //Error log
                m_Initialize = false;
                return tenStatus::nenError;
            }
        }


	}
}