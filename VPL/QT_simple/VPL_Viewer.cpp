#include "VPL_Viewer.hpp"
#include "VPL_FrameProcessor.hpp"
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
                _item = new QGraphicsPixmapItem;

                _scene = new QGraphicsScene(this);
                QGraphicsView *graphicsView = new QGraphicsView(_scene);
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
                connect(frameProc, SIGNAL(FrameBufferChanged(void*)), this, SLOT(setVideoData(void*)), Qt::DirectConnection);
            }

            void VPL_Viewer::setVideoData(void *buffer)
            {
                RW::tstBitStream *ptr = (RW::tstBitStream*)buffer;

#ifdef TEST
                static int count;
                WriteBufferToFile(ptr->pBuffer, ptr->u32Size, "Client_VPL", count);
#endif

                QImage img((uint8_t *)ptr->pBuffer, _width, _height, _format);

                QPixmap pix(QPixmap::fromImage(img));

                if (_count == 0)
                    _item = _scene->addPixmap(pix);
                else
                    _item->setPixmap(pix);

                //_scene->addEllipse(QRect(count, 0, 50, 50), QPen(Qt::red));
                SAFE_DELETE_ARRAY(ptr->pBuffer);
                SAFE_DELETE(ptr);

                _count++;
            }
        }
    }
}