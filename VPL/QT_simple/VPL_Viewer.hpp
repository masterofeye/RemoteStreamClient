#pragma once

#include <qmediaplayer.h>
#include <QtGui/QMovie>
#include <QtWidgets/QWidget>
#include <qbuffer.h>

QT_BEGIN_NAMESPACE
class QAbstractButton;
class QSlider;
class QLabel;
class QGraphicsVideoItem;
QT_END_NAMESPACE

namespace RW
{
    namespace VPL
    {
        namespace QT_SIMPLE
        {
            class VPL_FrameProcessor;
            class VPL_VideoItem;

#ifdef REMOTE_EXPORT
#define REMOTE_API __declspec(dllexport)
#else
#define REMOTE_API __declspec(dllimport)
#endif
            class  REMOTE_API VPL_Viewer : public QWidget
            {
                Q_OBJECT

            public:
                VPL_Viewer();
                ~VPL_Viewer();

            private:
                QMediaPlayer         mediaPlayer;
                QGraphicsVideoItem  *videoItem;
                QAbstractButton     *playButton;
                QSlider             *positionSlider;
                QLabel              *errorLabel;

                public slots:
                void setVideoData(QByteArray *qBuffer);
                void play();
                void connectToViewer(VPL_FrameProcessor *frameProc);

                private slots:
                void mediaStateChanged(QMediaPlayer::State state);
                void positionChanged(qint64 position);
                void durationChanged(qint64 duration);
                void setPosition(int position);
                void handleError();

            };
        }
    }
}