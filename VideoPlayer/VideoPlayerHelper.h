#pragma once

#include <QtWidgets>
#include <QtWidgets/QWidget>

namespace RW{
    namespace VPL{

        class VideoPlayerHelper : public QWidget
        {
        public:
            inline QIcon getQIcon(){ return style()->standardIcon(QStyle::SP_MediaPlay); }
        };

    }
}