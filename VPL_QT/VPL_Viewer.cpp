#include "VPL_Viewer.h"
#include <QtWidgets>
#include <QVideoSurfaceFormat>
#include <QGraphicsVideoItem>
#include <qbuffer.h>

namespace RW
{
    namespace VPL
    {
        VPL_Viewer::VPL_Viewer(QWidget *parent)
            : QWidget(parent)
            , mediaPlayer(0, QMediaPlayer::VideoSurface)
            , videoItem(0)
            , playButton(0)
            , positionSlider(0)
        {
            videoItem = new QGraphicsVideoItem;
            videoItem->setSize(QSizeF(640, 480));

            QGraphicsScene *scene = new QGraphicsScene(this);
            QGraphicsView *graphicsView = new QGraphicsView(scene);

            scene->addItem(videoItem);

            QAbstractButton *openButton = new QPushButton(tr("Connect..."));
            connect(openButton, SIGNAL(clicked()), this, SLOT(openFile()));

            playButton = new QPushButton;
            playButton->setEnabled(false);
            playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

            connect(playButton, SIGNAL(clicked()),
                this, SLOT(play()));

            positionSlider = new QSlider(Qt::Horizontal);
            positionSlider->setRange(0, 0);

            connect(positionSlider, SIGNAL(sliderMoved(int)),
                this, SLOT(setPosition(int)));

            QBoxLayout *controlLayout = new QHBoxLayout;
            controlLayout->setMargin(0);
            controlLayout->addWidget(openButton);
            controlLayout->addWidget(playButton);
            controlLayout->addWidget(positionSlider);

            QBoxLayout *layout = new QVBoxLayout;
            layout->addWidget(graphicsView);
            layout->addLayout(controlLayout);

            setLayout(layout);

            mediaPlayer.setVideoOutput(videoItem);
            connect(&mediaPlayer, SIGNAL(stateChanged(QMediaPlayer::State)),
                this, SLOT(mediaStateChanged(QMediaPlayer::State)));
            connect(&mediaPlayer, SIGNAL(positionChanged(qint64)), this, SLOT(positionChanged(qint64)));
            connect(&mediaPlayer, SIGNAL(durationChanged(qint64)), this, SLOT(durationChanged(qint64)));

        }

        void VPL_Viewer::setVideoData(QBuffer *qBuffer)
        {
            mediaPlayer.setMedia(QMediaContent(), qBuffer);
            playButton->setEnabled(true);
        }

        void VPL_Viewer::play()
        {
            switch (mediaPlayer.state()) {
            case QMediaPlayer::PlayingState:
                mediaPlayer.pause();
                break;
            default:
                mediaPlayer.play();
                break;
            }
        }

        void VPL_Viewer::mediaStateChanged(QMediaPlayer::State state)
        {
            switch (state) {
            case QMediaPlayer::PlayingState:
                playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
                break;
            default:
                playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
                break;
            }
        }

        void VPL_Viewer::positionChanged(qint64 position)
        {
            positionSlider->setValue(position);
        }

        void VPL_Viewer::durationChanged(qint64 duration)
        {
            positionSlider->setRange(0, duration);
        }

        void VPL_Viewer::setPosition(int position)
        {
            mediaPlayer.setPosition(position);
        }

    }
}