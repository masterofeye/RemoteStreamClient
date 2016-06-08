#pragma once

extern "C"{
#include "rwvx.h"
}
#include "Utils.h"
#include "spdlog\spdlog.h"

namespace RW
{
	namespace CORE
	{
        class Context;
        class Node;
        class AbstractModule;
		struct stControlStruct;
		typedef struct stControlStruct tstControlStruct;

        class REMOTE_API Kernel
		{
		private:
            vx_kernel       m_Kernel;
            vx_context      m_Context;
            bool			m_Initialize;
            std::string     m_KernelName;
            uint64_t        m_KernelEnum;
            uint8_t         m_CurrentParameterIndex;
            AbstractModule* m_AbstractModule;
            Node*            m_CurrentNode;
			RW::CORE::tstControlStruct *m_ControlStruct;
			std::shared_ptr<spdlog::logger> m_Logger;
		public:
			Kernel(Context *CurrentContext, RW::CORE::tstControlStruct *ControlStruct, std::string KernelName, uint64_t KernelEnum, uint8_t ParameterAmount, AbstractModule const *Module,std::shared_ptr<spdlog::logger> Logger);
			~Kernel();


			inline uint8_t CurrentParameterIndex() const { return m_CurrentParameterIndex; }
			inline std::string KernelName() const { return m_KernelName; }
			inline uint64_t KernelEnum() const { return m_KernelEnum; }
			inline void SetKernel(vx_kernel Kernel) { m_Kernel = Kernel; }
			inline bool IsInitialized() const { return m_Initialize; }
			inline RW::CORE::tstControlStruct *GetControlStruct(){ return m_ControlStruct; }
            void Kernel::SetParameter(int i, void* Value);
            inline void SetCurrentNode(RW::CORE::Node *CurrentNode){ m_CurrentNode = CurrentNode; }
            inline Node* Node(){ return m_CurrentNode; }

			vx_kernel operator()() const
            {
                if (m_Initialize)
                    return m_Kernel;
                return nullptr;
            }


			static vx_status KernelInitializeCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter);
            RW::tenStatus KernelInitialize(void* InitializeControlStruct);

            static vx_status KernelDeinitializeCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter);
            RW::tenStatus KernelDeinitialize(void* DeinitializeControlStruct);

            static vx_status KernelInputValidateCB(vx_node Node, vx_uint32  Index);

            static vx_status KernelOutputValidateCB(vx_node Node, vx_uint32  Index, vx_meta_format Meta);

            static vx_status KernelFncCB(vx_node Node, const vx_reference* Parameter, vx_uint32 NumberOfParameter);
            RW::tenStatus KernelFnc(void* ControlStruct);
        private:
            tenStatus AddKernel(uint8_t ParamterAmount);
		};

	}
}