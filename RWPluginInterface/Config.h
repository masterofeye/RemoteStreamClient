#pragma once

#include <QtCore\qobject.h>
#include <C:\tool\RemotePkg\OpenCV\build\install\include\opencv2\core\types.hpp>

#define TEST

namespace RW{

    struct sSimpleConfig{
        long lWidth;
        long lHeight;
        long lDisplayWidth;
        long lDisplayHeight;
        int iVideoType;
    };

    class cConfig{
    private:
        long m_lWidth;
        long m_lHeight;
        std::list<cv::Rect> m_lstRect;
    public:
        cConfig();
        ~cConfig();
        void SetWidth(long lWidth){ m_lWidth = lWidth; }
        void SetHeight(long lHeight){ m_lHeight = lHeight; }
        void SetRect(std::list<cv::Rect> lstRect){ m_lstRect = lstRect; }
        inline long GetDisplayWidth(){
            if (m_lstRect.size() == 0){
                return m_lWidth;
            }
            else{
                long lTmp = 0;
                for (std::list<cv::Rect>::iterator it = m_lstRect.begin(); it != m_lstRect.end(); it++){
                    lTmp += it->width;
                }
                return lTmp;
            }
        }
        inline long GetDisplayHeight(){
            if (m_lstRect.size() == 0){
                return m_lHeight;
            }
            else{
                long lTmp = 0;
                for (std::list<cv::Rect>::iterator it = m_lstRect.begin(); it != m_lstRect.end(); it++){
                    if (it->height > lTmp) lTmp = it->height;
                }
                return lTmp;
            }
        }
        inline int GetChannels()
        {
            int iChannels = 0;
#ifdef DEC_INTEL
            iChannels = 4;
#endif
#ifdef DEC_NVENC
            iChannels = 3;
#endif
            return iChannels;
        }
    };
}