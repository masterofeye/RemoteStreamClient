#pragma once

extern "C"{
#include "rwvx.h"
}
#include "Utils.h"
#include "spdlog\spdlog.h"
#include "spdlog\spdlog.h"

namespace RW
{
	namespace CORE
	{

        class REMOTE_API Context
		{
		private:
			vx_context  m_Context;
			bool		m_Initialize;
			vx_status   m_LastStatus;
            std::shared_ptr<spdlog::logger> m_Logger;
		public:
            Context(std::shared_ptr<spdlog::logger> Logger);
			~Context();

            vx_context operator() () const
			{
				if (m_Initialize)
					return m_Context;
				return nullptr;
			}

			inline  bool IsInitialized() const { return m_Initialize; }
            uint64_t Version();
            uint64_t Vendor();
			uint32_t UniqueKernels();
			tstKernelInfo* UniqueKernelTable(uint32_t AmountOfKernel);
			uint32_t ActiveModules();
			uint32_t ActiveReferences();
			std::string ImplementationName();
			size_t ExtentionSize();
			const std::string Extentions(size_t ExtentionSize);
			const size_t MaxConvolutionDimention();
			const size_t MaxOpticalFlowDimention();
			const tstBoderMode BorderMode();
			void SetBorderMode(tstBoderMode BorderMode);
		private:
			tenStatus CreateContext();
			tenStatus RegisterLogCallback();
			void LogCallback(vx_context Context, vx_reference Reference, vx_status, const vx_char String);
		};
	}
}


