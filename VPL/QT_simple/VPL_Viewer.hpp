#pragma once

#include <qmediaplayer.h>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE
class QGraphicsPixmapItem;
class QGraphicsScene;
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

            private:
                QGraphicsScene      *scene;
                QGraphicsPixmapItem *item;
                quint16             _width;
                quint16             _height;
                quint32             _count;
                QImage::Format      _format;    // for DEC::INTEL use QImage::Format::Format_RGBX8888; for DEC::NVEND use QImage::Format::Format_RGB888

            public:
                VPL_Viewer();
                ~VPL_Viewer();
                void setParams(quint16 width, quint16 height){ _width = width; _height = height; };
                void setImgType(QImage::Format format){ _format = format; };

                public slots:
                void setVideoData(void *buffer);
                void connectToViewer(VPL_FrameProcessor *frameProc);

            };
        }
    }
}