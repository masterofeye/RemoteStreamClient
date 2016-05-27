#pragma once

#include <qmediaplayer.h>

#include <QtGui/QMovie>
#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"
#include <QtWidgets>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE
class QAbstractButton;
class QSlider;
class QLabel;
QT_END_NAMESPACE

namespace RW{
    namespace VPL{

        typedef struct _EncodedBitStream
        {
            void *pBitStreamBuffer;
            uint32_t u32BitStreamSizeInBytes;

            _EncodedBitStream() : pBitStreamBuffer(nullptr), u32BitStreamSizeInBytes(0) {}
            ~_EncodedBitStream()
            {
                if (pBitStreamBuffer)
                {
                    delete pBitStreamBuffer;
                    pBitStreamBuffer = nullptr;
                }
            }
        }EncodedBitStream;

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
        }tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
        {
            EncodedBitStream *pstBitStream;
            stMyControlStruct() : pstBitStream(nullptr) {}
            ~stMyControlStruct()
            {
                if (pstBitStream)
                {
                    delete pstBitStream;
                    pstBitStream = nullptr;
                }
            }
        }tstMyControlStruct;

        typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
        }tstMyDeinitialiseControlStruct;

       class VideoPlayer : public RW::CORE::AbstractModule
        {
            Q_OBJECT

        public:
            explicit VideoPlayer(std::shared_ptr<spdlog::logger> Logger);
            ~VideoPlayer();
            virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
            virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
            virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

        private:
            void play();

            void mediaStateChanged(QMediaPlayer::State state);
            void positionChanged(qint64 position);
            void durationChanged(qint64 duration);
            void setPosition(int position);
            void handleError();

            QMediaPlayer      m_qmPlayer;
            QAbstractButton	  *m_qabPlay;
            QSlider           *m_qsPosition;
            QLabel            *m_qlError;
            QByteArray        *m_qbArray;
            QWidget           *m_pqWidget;
        };
    }
}

