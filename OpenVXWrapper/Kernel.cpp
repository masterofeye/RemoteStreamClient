#include "Kernel.h"
#include "Context.h"
#include "Plugin1.hpp"
namespace RW
{
	namespace CORE
	{
        Kernel::Kernel(Context *CurrentContext, std::string KernelName, uint64_t KernelEnum, AbstractModule const* Module)
            : m_KernelName(KernelName),
            m_KernelEnum(KernelEnum),
            m_Initialize(false),
            m_Context((*CurrentContext)())
		{

            m_AbstractModule = const_cast<AbstractModule*> (Module);
            AddKernel();
		}


		Kernel::~Kernel()
		{
		}



        vx_status VX_CALLBACK Kernel::KernelInitializeCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
        {
            vx_status status = VX_FAILURE;
            vx_array data;
            vx_parameter param[] = { vxGetParameterByIndex(Node, 0), vxGetParameterByIndex(Node, 1) };
            status = vxQueryParameter((vx_parameter)param[0], VX_PARAMETER_ATTRIBUTE_REF, &data, sizeof(data));
            if (status != VX_SUCCESS)
            {
                //TODO log error and specific return value
                return VX_FAILURE;
            }
            vx_size size;
            RW::VG::tstMyInitialiseControlStruct *test = nullptr;
            vxAccessArrayRange(data, 0, 1, &size, (void**)&test, VX_READ_AND_WRITE);

            status = vxQueryParameter((vx_parameter)param[1], VX_PARAMETER_ATTRIBUTE_REF, &data, sizeof(data));
            if (status != VX_SUCCESS)
            {
                //TODO log error and specific return value
                return VX_FAILURE;
            }
            Kernel *kernel = nullptr;
            vxAccessArrayRange(data, 0, 1, &size, (void**)&kernel, VX_READ_AND_WRITE);

            //Kernel* kernel = reinterpret_cast<Kernel*>(data.ptr);
            try
            {
                if (kernel != nullptr)
                {
                    kernel->KernelInitialize((RW::CORE::tstInitialiseControlStruct*) test);
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

            return status;
        }

        RW::tenStatus Kernel::KernelInitialize(void* ControlStruct)
        {
            tenStatus status = tenStatus::nenError;
            m_AbstractModule->Initialise((tstInitialiseControlStruct*)ControlStruct);
            return status;
        }

        vx_status  VX_CALLBACK Kernel::KernelDeinitializeCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
        {
            vx_status status = VX_FAILURE;
            rwvx_callback *data = 0;
            
            status = vxQueryParameter((vx_parameter)Parameter, VX_PARAMETER_ATTRIBUTE_REF, &data, sizeof(data));
            if (status != VX_SUCCESS)
            {
                //TODO log error and specific return value
                return VX_FAILURE;
            }
            Kernel* kernel = reinterpret_cast<Kernel*>(data->ptr);
            try
            {
                if (kernel != nullptr)
                {
                    kernel->KernelDeinitialize();
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

            return status;
        }
        
        RW::tenStatus Kernel::KernelDeinitialize()
        {
            tenStatus status = tenStatus::nenError;

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
            rwvx_callback *data = 0;

            status = vxQueryParameter((vx_parameter)Parameter, VX_PARAMETER_ATTRIBUTE_REF, &data, sizeof(data));
            if (status != VX_SUCCESS)
            {
                //TODO log error and specific return value
                return VX_FAILURE;
            }

            Kernel* kernel = reinterpret_cast<Kernel*>(data->ptr);
            try
            {
                if (kernel != nullptr)
                {
                    kernel->KernelFnc();
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

            return status;
        }

        tenStatus Kernel::KernelFnc()
        {
            tenStatus status = tenStatus::nenError;

            return status;
        }

        tenStatus Kernel::AddKernel()
        {
            if (m_Context == nullptr)
            {
                //Todo Error
                return tenStatus::nenError;
            }

            vx_kernel kernel = vxAddKernel(m_Context, this->KernelName().c_str(), this->KernelEnum(), this->KernelFncCB, 2, this->KernelInputValidateCB, this->KernelOutputValidateCB, this->KernelInitializeCB, this->KernelDeinitializeCB);

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