/****************************************************************************
**
** Copyright (C) 2015 The Qt Company Ltd.
** Contact: http://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "videoplayer.hpp"

#include <QtWidgets>
#include <qvideowidget.h>
#include <qvideosurfaceformat.h>
#include "qobject.h"

namespace RW
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
		: QWidget(0)
		, m_qmPlayer(0, QMediaPlayer::VideoSurface)
		, m_qabPlay(0)
		, m_qsPosition(0)
		, m_qlError(0)
		, RW::CORE::AbstractModule(Logger)
	{
		QVideoWidget *videoWidget = new QVideoWidget;
		m_qbArray = new QByteArray();

		QAbstractButton *openButton = new QPushButton(tr("Open..."));
		QObject::connect(openButton, SIGNAL(clicked()), videoWidget, SLOT(openFile()));

		m_qabPlay = new QPushButton;
		m_qabPlay->setEnabled(false);
		m_qabPlay->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

		QObject::connect(m_qabPlay, SIGNAL(clicked()),
			videoWidget, SLOT(play()));

		m_qsPosition = new QSlider(Qt::Horizontal);
		m_qsPosition->setRange(0, 0);

		QObject::connect(m_qsPosition, SIGNAL(sliderMoved(int)),
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

		setLayout(layout);

		m_qmPlayer.setVideoOutput(videoWidget);
		QObject::connect(&m_qmPlayer, SIGNAL(stateChanged(QMediaPlayer::State)),
			videoWidget, SLOT(mediaStateChanged(QMediaPlayer::State)));
		QObject::connect(&m_qmPlayer, SIGNAL(positionChanged(qint64)), videoWidget, SLOT(positionChanged(qint64)));
		QObject::connect(&m_qmPlayer, SIGNAL(durationChanged(qint64)), videoWidget, SLOT(durationChanged(qint64)));
		QObject::connect(&m_qmPlayer, SIGNAL(error(QMediaPlayer::Error)), videoWidget, SLOT(handleError()));
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

		resize(320, 240);
		show();

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

		m_qbArray->data = (char*)data->stBitStream.pBitStreamBuffer;
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

		QString fileName = QFileDialog::getOpenFileName(this, tr("Open Movie"), QDir::homePath());

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
			m_qabPlay->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
			break;
		default:
			m_qabPlay->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
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