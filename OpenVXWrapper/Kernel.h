#pragma once
#include <QObject>

#include <spdlog.h>

extern "C" {
#include <VX\vx.h>
}

//Own Headers
#include "..\RWPluginInterface\AbstractModule.h"
#include "openvxwrapper_global.h"


namespace RW
{
	namespace CORE
	{
		class OPENVXWRAPPER_EXPORT Kernel : public QObject
		{
		private:
			vx_kernel       m_Kernel;
			vx_context      m_Context;
			bool			m_Initialize;
			QString         m_KernelName;
			uint64_t        m_KernelEnum;

			AbstractModule* m_AbstractModule;

			std::shared_ptr<spdlog::logger> m_Logger;
		public:
			Kernel();
			~Kernel();

			/******************************************************************************************************************
			@autor Ivo Kunadt
			@brief
			@param
			@return
			********************************************************************************************************************/
			vx_status KernelPuplishWrapper(vx_context context);

			/******************************************************************************************************************
			@autor Ivo Kunadt
			@brief
			@param
			@return
			********************************************************************************************************************/
			vx_status KernelDeinitializeWrapper(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter);

			/******************************************************************************************************************
			@autor Ivo Kunadt
			@brief
			@param
			@return
			********************************************************************************************************************/
			vx_status KernelInitializeWrapper(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter);

			/******************************************************************************************************************
			@autor Ivo Kunadt
			@brief
			@param
			@return
			********************************************************************************************************************/
			vx_status KernelFunctionWrapper(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter);

			/******************************************************************************************************************
			@autor Ivo Kunadt
			@brief
			@param
			@return
			********************************************************************************************************************/
			vx_status KernelInputValidateWrapper(vx_node Node, vx_uint32 Index);

			/******************************************************************************************************************
			@autor Ivo Kunadt
			@brief
			@param
			@return
			********************************************************************************************************************/
			vx_status KernelOutputValidateWrapper(vx_node Node, vx_uint32 Index, vx_meta_format Meta);

		};
	}
}
