#pragma once

#include "AbstractModule.hpp"


QT_BEGIN_NAMESPACE
class QBuffer;
QT_END_NAMESPACE

namespace RW{
    namespace VPL{

        class VPL_Viewer;

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

        public slots:
            QBuffer *GetNewFrameBuffer(void){ return m_pqFrameBuffer; }

        signals:
            void FrameBufferChanged(QBuffer* pBuffer);

        private:
            QBuffer     *m_pqFrameBuffer;
            VPL_Viewer  *m_pqViewer;

        public:
            explicit VPL_FrameProcessor(std::shared_ptr<spdlog::logger> Logger);
            ~VPL_FrameProcessor();
            virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
            virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
            virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

            VPL_FrameProcessor *GetObject(){ return this; }

        };
    }
}

