#include "Kernel.h"



namespace RW
{
	namespace CORE
	{
		Kernel::Kernel()
		{
		}


		Kernel::~Kernel()
		{
		}

		vx_status Kernel::KernelPuplishWrapper(vx_context context)
		{
			vx_status status = VX_FAILURE;
			return status;
		}

		vx_status Kernel::KernelDeinitializeWrapper(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
		{
			vx_status status = VX_FAILURE;
			return status;
		}

		vx_status Kernel::KernelInitializeWrapper(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
		{
			vx_status status = VX_FAILURE;
			return status;
		}

		vx_status Kernel::KernelFunctionWrapper(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter)
		{
			vx_status status = VX_FAILURE;
			return status;
		}

		vx_status Kernel::KernelInputValidateWrapper(vx_node Node, vx_uint32 Index)
		{
			vx_status status = VX_FAILURE;
			return status;
		}

		vx_status Kernel::KernelOutputValidateWrapper(vx_node Node, vx_uint32 Index, vx_meta_format Meta)
		{
			vx_status status = VX_FAILURE;
			return status;
		}
	}
}