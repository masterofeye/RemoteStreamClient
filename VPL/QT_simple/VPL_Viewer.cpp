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
                _count = 0;
            }

            VPL_Viewer::~VPL_Viewer()
            {
                printf("Destructor VPL_Viewer");
            }

            void VPL_Viewer::connectToViewer(VPL_FrameProcessor *frameProc)
            {
                //To enable asynchronous threads we have to select QueuedConnection. This will create a copy of the parameter. 
                connect(frameProc, SIGNAL(FrameBufferChanged(void*)), this, SLOT(setVideoData(void*)), Qt::DirectConnection);
            }

            void VPL_Viewer::setVideoData(void *buffer)
            {
                RW::tstBitStream* ptr = (RW::tstBitStream*)buffer;
                QImage img(ptr->pBuffer, _width, _height, _format);

                QPixmap pix(QPixmap::fromImage(img));

                if (_count == 0)
                    item = scene->addPixmap(pix);
                else
                    item->setPixmap(pix);

                //FILE *pFile;
                //fopen_s(&pFile, "c:\\dummy\\outViewer.raw", "wb");
                //fwrite(buffer, 1,_width*_height*3, pFile);
                //fclose(pFile);

                //scene->addEllipse(QRect(count, 0, 50, 50), QPen(Qt::red));
                SAFE_DELETE(ptr);

                _count++;
            }
        }
    }
}