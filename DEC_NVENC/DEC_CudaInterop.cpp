

#include "DEC_CudaInterop.hpp"
#include "..\VPL_QT\VPL_FrameProcessor.hpp"

namespace RW
{
	namespace DEC
	{
		void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
		{
			switch (SubModuleType)
			{
			case RW::CORE::tenSubModule::nenPlayback_Simple:
			{
				RW::VPL::tstMyControlStruct *data = static_cast<RW::VPL::tstMyControlStruct*>(*Data);
				data->pstBitStream = this->pOutput;
				break;
			}
			default:
				break;
			}
		}

		DEC_CudaInterop::DEC_CudaInterop(std::shared_ptr<spdlog::logger> Logger) : RW::CORE::AbstractModule(Logger)
		{
			m_pNvDecodeD3D9 = new CNvDecodeD3D9(Logger);
		}

		DEC_CudaInterop::~DEC_CudaInterop()
		{
			if (m_pNvDecodeD3D9)
			{
				delete m_pNvDecodeD3D9;
				m_pNvDecodeD3D9 = nullptr;
			}
		}

		CORE::tstModuleVersion DEC_CudaInterop::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

		CORE::tenSubModule DEC_CudaInterop::SubModulType()
		{
			return CORE::tenSubModule::nenDecoder_INTEL;
		}

		tenStatus DEC_CudaInterop::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
		{

			tenStatus enStatus = tenStatus::nenSuccess;
			m_Logger->debug("Initialise nenDecoder_NVENC");
#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

			stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

			if (!data)
			{
				m_Logger->error("DEC_CudaInterop::Initialise: Data of tstMyInitialiseControlStruct is empty!");
				return tenStatus::nenError;
			}


#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "DEC_CudaInterop::Initialise: Time to Initialise for nenDecoder_INTEL module: " << (RW::CORE::HighResClock::diffMilli(t1, t2).count()) << "ms.";
#endif
			return enStatus;
		}

		tenStatus DEC_CudaInterop::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			m_Logger->debug("DoRender nenDecoder_NVENC");
#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
			if (!data)
			{
				m_Logger->error("DEC_CudaInterop::DoRender: Data of stMyControlStruct is empty!");
				return tenStatus::nenError;
			}
			if (!data->pOutput)
			{
				m_Logger->error("DEC_CudaInterop::DoRender: pOutput of stMyControlStruct is empty!");
				return tenStatus::nenError;
			}

			for (;;)
			{
				
			}
#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "DEC_CudaInterop::DoRender: Time to DoRender for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
			return enStatus;
		}

		tenStatus DEC_CudaInterop::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{
			m_Logger->debug("Deinitialise nenDecoder_NVENC");
#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
			RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
			m_Logger->trace() << "DEC_CudaInterop::Deinitialise: Time to Deinitialise for nenDecoder_INTEL module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
			return tenStatus::nenSuccess;
		}

	}

}


