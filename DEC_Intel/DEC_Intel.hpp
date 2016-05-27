#pragma once

#include "mfx_samples_config.h"

#include <sstream>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"
#include "pipeline_decode.h"

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW{
    namespace DEC{

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
            sInputParams inputParams;
        }tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
        {
            BitStream *pOutput;
            BitStream *pstEncodedStream;

            stMyControlStruct() : pOutput(nullptr), pstEncodedStream(nullptr){}
            ~stMyControlStruct()
            {
                if (pOutput)
                {
                    delete pOutput;
                    pOutput = nullptr;
                }
                if (pstEncodedStream)
                {
                    delete pstEncodedStream;
                    pstEncodedStream = nullptr;
                }
            }
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
            CDecodingPipeline   m_Pipeline; // pipeline for decoding, includes input file reader, decoder and output file writer

        };
    }
}