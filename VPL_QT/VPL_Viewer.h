#pragma once

#include <QMediaPlayer>
#include <QWidget>

class QAbstractButton;
class QBuffer;
class QSlider;
class QGraphicsVideoItem;

namespace RW
{
    namespace VPL
    {
        class VPL_Viewer : public QWidget
        {
        public:
            VPL_Viewer(QWidget *parent = 0);
            ~VPL_Viewer(){};

            public slots:
            void setVideoData(QBuffer *qBuffer);
            void play();

            private slots:
            void mediaStateChanged(QMediaPlayer::State state);
            void positionChanged(qint64 position);
            void durationChanged(qint64 duration);
            void setPosition(int position);

        private:
            QMediaPlayer mediaPlayer;
            QGraphicsVideoItem *videoItem;
            QAbstractButton *playButton;
            QSlider *positionSlider;
        };

    }
}