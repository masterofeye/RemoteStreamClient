#include "VPL_Viewer.hpp"
#include "VPL_FrameProcessor.hpp"
#include <QtWidgets>
#include <QVideoSurfaceFormat>
#include <QGraphicsVideoItem>

namespace RW
{
    namespace VPL
    {
        namespace QT_SIMPLE
        {
            VPL_Viewer::VPL_Viewer()
                : QWidget(0)
                , mediaPlayer(0, QMediaPlayer::VideoSurface)
                , videoItem(0)
                , playButton(0)
                , positionSlider(0)
                , errorLabel(0)
            {
                videoItem = new QGraphicsVideoItem;

                QGraphicsScene *scene = new QGraphicsScene(this);
                QGraphicsView *graphicsView = new QGraphicsView(scene);

                scene->addItem(videoItem);

                playButton = new QPushButton;
                playButton->setEnabled(false);
                playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));

                connect(playButton, SIGNAL(clicked()),
                    this, SLOT(play()));

                positionSlider = new QSlider(Qt::Horizontal);
                positionSlider->setRange(0, 0);

                connect(positionSlider, SIGNAL(sliderMoved(int)),
                    this, SLOT(setPosition(int)));

                errorLabel = new QLabel;
                errorLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

                QBoxLayout *controlLayout = new QHBoxLayout;
                controlLayout->setMargin(0);
                controlLayout->addWidget(playButton);
                controlLayout->addWidget(positionSlider);

                QBoxLayout *layout = new QVBoxLayout;
                layout->addWidget(graphicsView);
                layout->addLayout(controlLayout);
                layout->addWidget(errorLabel);

                setLayout(layout);

                mediaPlayer.setVideoOutput(videoItem);
                connect(&mediaPlayer, SIGNAL(stateChanged(QMediaPlayer::State)),
                    this, SLOT(mediaStateChanged(QMediaPlayer::State)));
                connect(&mediaPlayer, SIGNAL(positionChanged(qint64)), this, SLOT(positionChanged(qint64)));
                connect(&mediaPlayer, SIGNAL(durationChanged(qint64)), this, SLOT(durationChanged(qint64)));
                connect(&mediaPlayer, SIGNAL(error(QMediaPlayer::Error)), this, SLOT(handleError()));

            }

            VPL_Viewer::~VPL_Viewer()
            {
                delete videoItem;
                videoItem = nullptr;
                delete playButton;
                playButton = nullptr;
                delete positionSlider;
                positionSlider = nullptr;
                delete errorLabel;
                errorLabel = nullptr;
            }

            void VPL_Viewer::connectToViewer(VPL_FrameProcessor *frameProc)
            {
                connect(frameProc, SIGNAL(FrameBufferChanged(QByteArray*)), this, SLOT(setVideoData(QByteArray*)), Qt::AutoConnection);
            }

            void VPL_Viewer::setVideoData(QByteArray *qBuffer)
            {
                QBuffer buffer(qBuffer);
                printf("----------------------VPL_Viewer::setVideoData executing!-------------------");
                mediaPlayer.setMedia(QMediaContent(), &buffer);
                playButton->setEnabled(true);
                if (qBuffer)
                {
                    delete qBuffer;
                    qBuffer = nullptr;
                }
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
            void VPL_Viewer::handleError()
            {
                playButton->setEnabled(false);
                errorLabel->setText("Error: " + mediaPlayer.errorString());
            }
        }
    }
}