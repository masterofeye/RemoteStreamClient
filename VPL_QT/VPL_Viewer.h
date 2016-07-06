#pragma once

#include <qmediaplayer.h>
#include <QtGui/QMovie>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE
class QAbstractButton;
class QBuffer;
class QSlider;
class QLabel;
class QGraphicsVideoItem;
QT_END_NAMESPACE

namespace RW
{
    namespace VPL
    {
        class VPL_FrameProcessor;

        class VPL_Viewer : public QWidget
        {
            Q_OBJECT

        public:
            VPL_Viewer(VPL_FrameProcessor *pFrameProc);
            ~VPL_Viewer();

        private:
            VPL_FrameProcessor  *frameProc;
            QMediaPlayer         mediaPlayer;
            QGraphicsVideoItem  *videoItem;
            QAbstractButton     *playButton;
            QSlider             *positionSlider;
            QLabel              *errorLabel;

        public slots:
            void setVideoData(QBuffer *qBuffer);
            void play();

        private slots:
            void mediaStateChanged(QMediaPlayer::State state);
            void positionChanged(qint64 position);
            void durationChanged(qint64 duration);
            void setPosition(int position);
            void handleError();

        };
    }
}