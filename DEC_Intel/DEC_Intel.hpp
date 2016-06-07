#pragma once

#include <sstream>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"
#include "DEC_inputs.h"

namespace RW{
    namespace DEC{

		class CDecodingPipeline;

		typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
			tstInputParams *inputParams;
        }tstMyInitialiseControlStruct;

        typedef struct REMOTE_API stMyControlStruct : public CORE::tstControlStruct
        {
            tstBitStream *pOutput;
            tstBitStream *pstEncodedStream;
            tstBitStream *pPayload;
            void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
        }tstMyControlStruct;

        typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
        }tstMyDeinitialiseControlStruct;

        class DEC_Intel : public RW::CORE::AbstractModule
        {
            Q_OBJECT

        public:

            explicit DEC_Intel(std::shared_ptr<spdlog::logger> Logger);
            ~DEC_Intel();
            virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
            virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
            virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

        private:
            CDecodingPipeline   *m_pPipeline; // pipeline for decoding

        };
    }
}