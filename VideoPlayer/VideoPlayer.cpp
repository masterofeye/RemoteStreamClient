
#include "VideoPlayer.hpp"

#include <qvideowidget.h>
#include <qvideosurfaceformat.h>
#include "qobject.h"

namespace RW
{
	namespace VPL
	{
		CORE::tstModuleVersion VideoPlayer::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

		CORE::tenSubModule VideoPlayer::SubModulType()
		{
			return CORE::tenSubModule::nenPlayback_Simple;
		}


		VideoPlayer::VideoPlayer(std::shared_ptr<spdlog::logger> Logger)
			: RW::CORE::AbstractModule(Logger)
			, m_qmPlayer(0, QMediaPlayer::VideoSurface)
			, m_qabPlay(0)
			, m_qsPosition(0)
			, m_qlError(0)
		{
				QVideoWidget *videoWidget = new QVideoWidget;
				m_qbArray = new QByteArray();

				QAbstractButton *openButton = new QPushButton(tr("Open..."));
				connect(openButton, SIGNAL(clicked()), videoWidget, SLOT(openFile()));

				m_pVPH = new VideoPlayerHelper();
				m_qabPlay = new QPushButton;
				m_qabPlay->setEnabled(false);
				m_qabPlay->setIcon(m_pVPH->getQIcon());

				connect(m_qabPlay, SIGNAL(clicked()),
					videoWidget, SLOT(play()));

				m_qsPosition = new QSlider(Qt::Horizontal);
				m_qsPosition->setRange(0, 0);

				connect(m_qsPosition, SIGNAL(sliderMoved(int)),
					videoWidget, SLOT(setPosition(int)));

				m_qlError = new QLabel;
				m_qlError->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

				QBoxLayout *controlLayout = new QHBoxLayout;
				controlLayout->setMargin(0);
				controlLayout->addWidget(openButton);
				controlLayout->addWidget(m_qabPlay);
				controlLayout->addWidget(m_qsPosition);

				QBoxLayout *layout = new QVBoxLayout;
				layout->addWidget(videoWidget);
				layout->addLayout(controlLayout);
				layout->addWidget(m_qlError);

				m_pVPH->setLayout(layout);

				m_qmPlayer.setVideoOutput(videoWidget);
				connect(&m_qmPlayer, SIGNAL(stateChanged(QMediaPlayer::State)),
					videoWidget, SLOT(mediaStateChanged(QMediaPlayer::State)));
				connect(&m_qmPlayer, SIGNAL(positionChanged(qint64)), videoWidget, SLOT(positionChanged(qint64)));
				connect(&m_qmPlayer, SIGNAL(durationChanged(qint64)), videoWidget, SLOT(durationChanged(qint64)));
				connect(&m_qmPlayer, SIGNAL(error(QMediaPlayer::Error)), videoWidget, SLOT(handleError()));
			}

		VideoPlayer::~VideoPlayer()
		{
		}

		tenStatus VideoPlayer::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
			//if (data == NULL)
			//{
			//	m_Logger->error("Initialise: Data of tstMyInitialiseControlStruct is empty!");
			//	enStatus = tenStatus::nenError;
			//	return enStatus;
			//}

			m_pVPH->resize(320, 240);
			m_pVPH->show();

			m_Logger->debug("Initialise");
			return enStatus;
		}
		tenStatus VideoPlayer::DoRender(CORE::tstControlStruct * ControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;

			stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
			if (data == NULL)
			{
				m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
				enStatus = tenStatus::nenError;
				return enStatus;
			}

			m_qbArray->setRawData((char*)data->stBitStream.pBitStreamBuffer, data->stBitStream.u32BitStreamSizeInBytes);
			m_qmPlayer.setMedia(QUrl::fromEncoded(*m_qbArray));
			m_qabPlay->setEnabled(true);
			m_qmPlayer.play();

			m_Logger->debug("DoRender");
			return enStatus;
		}

		tenStatus VideoPlayer::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
		{
			tenStatus enStatus = tenStatus::nenSuccess;
			stMyDeinitialiseControlStruct* data = static_cast<stMyDeinitialiseControlStruct*>(DeinitialiseControlStruct);
			//if (data == NULL)
			//{
			//	m_Logger->error("Deinitialise: Data of stMyDeinitialiseControlStruct is empty!");
			//	enStatus = tenStatus::nenError;
			//	return enStatus;
			//}

			m_Logger->debug("Deinitialise");
			return enStatus;
		}

		void VideoPlayer::openFile()
		{
			m_qlError->setText("");

			QString fileName = QFileDialog::getOpenFileName(m_pVPH, tr("Open Movie"), QDir::homePath());

			if (!fileName.isEmpty()) {
				m_qmPlayer.setMedia(QUrl::fromLocalFile(fileName));
				m_qabPlay->setEnabled(true);
			}
		}

		void VideoPlayer::play()
		{
			switch (m_qmPlayer.state()) {
			case QMediaPlayer::PlayingState:
				m_qmPlayer.pause();
				break;
			default:
				m_qmPlayer.play();
				break;
			}
		}

		void VideoPlayer::mediaStateChanged(QMediaPlayer::State state)
		{
			switch (state) {
			case QMediaPlayer::PlayingState:
				m_qabPlay->setIcon(m_pVPH->style()->standardIcon(QStyle::SP_MediaPause));
				break;
			default:
				m_qabPlay->setIcon(m_pVPH->getQIcon());
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
			m_qabPlay->setEnabled(false);
			m_qlError->setText("Error: " + m_qmPlayer.errorString());
		}
	}
}