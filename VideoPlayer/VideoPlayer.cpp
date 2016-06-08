
#include "VideoPlayer.hpp"
#include "qbuffer.h"
#include "qboxlayout.h"
#include "qpushbutton.h"

#include <qvideowidget.h>
#include <qvideosurfaceformat.h>
#include "qobject.h"
#include "qslider.h"
#include "qlabel.h"
#include "qwidget.h"
#include "qstyle.h"

namespace RW
{
    namespace VPL
    {
        void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
        {
        }

        VideoPlayer::VideoPlayer(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
            , m_qmPlayer(0, QMediaPlayer::VideoSurface)
            , m_qaBtnPlay(0)
            , m_qaBtnStop(0)
            , m_qsPosition(0)
            , m_qlError(0)
        {
                

                m_pqWidget = new QWidget();
                m_qaBtnPlay = new QPushButton;
                m_qaBtnStop = new QPushButton;
                m_qsPosition = new QSlider(Qt::Horizontal);
                m_qlError = new QLabel;

                QApplication app();
            }

        VideoPlayer::~VideoPlayer()
        {
            if (m_qaBtnPlay != nullptr)
            {
                delete m_qaBtnPlay;
                m_qaBtnPlay = nullptr;
            }
            if (m_qaBtnStop != nullptr)
            {
                delete m_qaBtnStop;
                m_qaBtnStop = nullptr;
            }
            if (m_qsPosition != nullptr)
            {
                delete m_qsPosition;
                m_qsPosition = nullptr;
            }
            if (m_qlError != nullptr)
            {
                delete m_qlError;
                m_qlError = nullptr;
            }
            if (m_pqWidget != nullptr)
            {
                delete m_pqWidget;
                m_pqWidget = nullptr;
            }
        }

        CORE::tstModuleVersion VideoPlayer::ModulVersion() {
            CORE::tstModuleVersion version = { 0, 1 };
            return version;
        }

        CORE::tenSubModule VideoPlayer::SubModulType()
        {
            return CORE::tenSubModule::nenPlayback_Simple;
        }

        tenStatus VideoPlayer::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
        {
            tenStatus enStatus = tenStatus::nenSuccess;

            m_Logger->debug("Initialise nenPlayback_Simple");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

            //stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);

            m_qsPosition->setRange(0, 0);
            m_qaBtnPlay->setEnabled(false);
            m_qaBtnStop->setEnabled(false);
            m_qaBtnPlay->setIcon(m_pqWidget->style()->standardIcon(QStyle::SP_MediaPlay));
            m_qaBtnPlay->setIcon(m_pqWidget->style()->standardIcon(QStyle::SP_MediaStop));
            m_qlError->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

            QVideoWidget *pVideoWidget = new QVideoWidget;

            connect(m_qaBtnStop, SIGNAL(clicked()),
                this, SLOT(stop()));

            connect(m_qaBtnPlay, SIGNAL(clicked()),
                this, SLOT(play()));

            connect(m_qsPosition, SIGNAL(sliderMoved(int)),
                this, SLOT(setPosition(int)));

            QBoxLayout *controlLayout = new QHBoxLayout;
            controlLayout->setMargin(0);
            controlLayout->addWidget(m_qaBtnStop);
            controlLayout->addWidget(m_qaBtnPlay);
            controlLayout->addWidget(m_qsPosition);

            QBoxLayout *layout = new QVBoxLayout;
            layout->addWidget(pVideoWidget);
            layout->addLayout(controlLayout);
            layout->addWidget(m_qlError);

            m_pqWidget->setLayout(layout);

            m_qmPlayer.setVideoOutput(pVideoWidget);
            connect(&m_qmPlayer, SIGNAL(stateChanged(QMediaPlayer::State)),
                this, SLOT(mediaStateChanged(QMediaPlayer::State)));
            connect(&m_qmPlayer, SIGNAL(positionChanged(qint64)), this, SLOT(positionChanged(qint64)));
            connect(&m_qmPlayer, SIGNAL(durationChanged(qint64)), this, SLOT(durationChanged(qint64)));
            connect(&m_qmPlayer, SIGNAL(error(QMediaPlayer::Error)), this, SLOT(handleError()));

            m_pqWidget->resize(320, 240);
            m_pqWidget->show();

            //if (pVideoWidget)
            //{
            //    delete pVideoWidget;
            //    pVideoWidget = nullptr;
            //}
            //if (controlLayout)
            //{
            //    delete controlLayout;
            //    controlLayout = nullptr;
            //}
            //if (layout)
            //{
            //    delete layout;
            //    layout = nullptr;
            //}

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to Initialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
        }
        tenStatus VideoPlayer::DoRender(CORE::tstControlStruct * ControlStruct)
        {
            tenStatus enStatus = tenStatus::nenSuccess;
            m_Logger->debug("DoRender nenPlayback_Simple");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

            stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
            if (data == nullptr)
            {
                m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                enStatus = tenStatus::nenError;
                return enStatus;
            }
            QByteArray *pqbArray = new QByteArray();
            pqbArray->setRawData((char*)data->pstBitStream->pBuffer, data->pstBitStream->u32Size);
            QBuffer qBuffer(pqbArray);
            m_qmPlayer.setMedia(QMediaContent(), &qBuffer);
            m_qaBtnPlay->setEnabled(true);
            m_qaBtnStop->setEnabled(true);
            m_qmPlayer.play();
            
            if ((m_qmPlayer.mediaStatus() == m_qmPlayer.EndOfMedia) 
                || (m_qmPlayer.mediaStatus() == m_qmPlayer.InvalidMedia)
                || (m_qmPlayer.mediaStatus() == m_qmPlayer.NoMedia)
                || (m_qmPlayer.mediaStatus() == m_qmPlayer.UnknownMediaStatus)
                || (m_qmPlayer.error() != m_qmPlayer.NoError) 
                || (m_qmPlayer.state() == m_qmPlayer.StoppedState))
            {
                if (pqbArray)
                {
                    delete pqbArray;
                    pqbArray = nullptr;
                }
            }
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to DoRender for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
        }

        tenStatus VideoPlayer::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            m_Logger->debug("Deinitialise nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to Deinitialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
        }

        void VideoPlayer::play()
        {
            switch (m_qmPlayer.state()) {
            case QMediaPlayer::PlayingState:
                m_qmPlayer.pause();
                m_qaBtnStop->setEnabled(true);
                break;
            default:
                m_qmPlayer.play();
                m_qaBtnStop->setEnabled(true);
                break;
            }
        }

        void VideoPlayer::stop()
        {
            m_qmPlayer.stop();
            m_qaBtnStop->setEnabled(false);
            //m_qmPlayer.state = m_qmPlayer.StoppedState;
        }

        void VideoPlayer::mediaStateChanged(QMediaPlayer::State state)
        {
            switch (state) {
            case QMediaPlayer::PlayingState:
                m_qaBtnPlay->setIcon(m_pqWidget->style()->standardIcon(QStyle::SP_MediaPause));
                break;
            case QMediaPlayer::StoppedState:
                //m_qaBtnStop->
                break;
            case QMediaPlayer::PausedState:
                m_qaBtnPlay->setIcon(m_pqWidget->style()->standardIcon(QStyle::SP_MediaPlay));
                break;
            default:
                break;
            }
        }

        void VideoPlayer::positionChanged(qint64 position)
        {
            m_qsPosition->setValue(position);
        }

        void VideoPlayer::durationChanged(qint64 duration)
        {
            m_qsPosition->setRange(0, duration);
        }

        void VideoPlayer::setPosition(int position)
        {
            m_qmPlayer.setPosition(position);
        }

        void VideoPlayer::handleError()
        {
            m_qaBtnPlay->setEnabled(false);
            m_qaBtnStop->setEnabled(false);
            m_Logger->error() << m_qmPlayer.errorString().toStdString().c_str();
        }
    }
}