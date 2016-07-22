#pragma once

#include "DEC_NVENC_inputs.h"
#include "AbstractModule.hpp"
#include "NvDecodeD3D9.h"

namespace RW
{
	namespace DEC
	{

		class CNvDecodeD3D9;

		typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
		{
			tstInputParams *inputParams;
		}tstMyInitialiseControlStruct;

		typedef struct stMyControlStruct : public CORE::tstControlStruct
		{
			tstBitStream *pOutput;
			tstBitStream *pstEncodedStream;
			tstBitStream *pPayload;
			REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
		}tstMyControlStruct;

		typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
		{
		}tstMyDeinitialiseControlStruct;

		class DEC_CudaInterop : public RW::CORE::AbstractModule
		{
			Q_OBJECT

		public:

			explicit DEC_CudaInterop(std::shared_ptr<spdlog::logger> Logger);
			~DEC_CudaInterop();
			virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
			virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
			virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

		private:
			CNvDecodeD3D9   *pNvDecodeD3D9;

		};
	}
}