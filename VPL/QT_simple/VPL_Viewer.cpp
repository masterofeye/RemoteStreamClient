#include "VPL_Viewer.hpp"
#include "VPL_FrameProcessor.hpp"
//#include <QtWidgets>
#include <QGraphicsItem>
#include <QGraphicsView>
#include <QBoxLayout>
#include "opencv2/opencv.hpp"

namespace RW
{
    namespace VPL
    {
        namespace QT_SIMPLE
        {
            VPL_Viewer::VPL_Viewer()
                : QWidget(0)
            {
                item = new QGraphicsPixmapItem;

                scene = new QGraphicsScene(this);
                QGraphicsView *graphicsView = new QGraphicsView(scene);
                //graphicsView->scale(1, -1); 

                QBoxLayout *controlLayout = new QHBoxLayout;
                controlLayout->setMargin(0);

                QBoxLayout *layout = new QVBoxLayout;
                layout->addWidget(graphicsView);
                layout->addLayout(controlLayout);

                setLayout(layout);

                _width = 0;
                _height = 0;
                count = 0;
            }

            VPL_Viewer::~VPL_Viewer()
            {
            }

            void VPL_Viewer::connectToViewer(VPL_FrameProcessor *frameProc)
            {
                //To enable asynchronous threads we have to select QueuedConnection. This will create a copy of the parameter. 
                connect(frameProc, SIGNAL(FrameBufferChanged(uchar*)), this, SLOT(setVideoData(uchar*)), Qt::QueuedConnection);
            }

            void VPL_Viewer::setVideoData(uchar *buffer)
            {
                QImage img(buffer, _width, _height, QImage::Format::Format_RGBX8888);

                QPixmap pix(QPixmap::fromImage(img));

                if (count == 0)
                    item = scene->addPixmap(pix);
                else
                    item->setPixmap(pix);

                //scene->addEllipse(QRect(count, 0, 50, 50), QPen(Qt::red));
                count++;

                SAFE_DELETE(buffer);
            }
        }
    }
}