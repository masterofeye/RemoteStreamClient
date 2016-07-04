#pragma once

#include <qmediaplayer.h>
#include "AbstractModule.hpp"
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE
class QAbstractButton;
class QSlider;
class QLabel;
QT_END_NAMESPACE

namespace RW{
    namespace VPL{

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
        }tstMyInitialiseControlStruct;

        typedef struct REMOTE_API stMyControlStruct : public CORE::tstControlStruct
        {
            tstBitStream *pstBitStream;
            uint64_t    TimeStamp;
            void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
        }tstMyControlStruct;

        typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
        }tstMyDeinitialiseControlStruct;

        class VPL_FrameProcessor : public RW::CORE::AbstractModule
        {
            Q_OBJECT

        public:
            explicit VPL_FrameProcessor(std::shared_ptr<spdlog::logger> Logger);
            ~VPL_FrameProcessor();
            virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
            virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
            virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

        private:
        };
    }
}

