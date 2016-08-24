//
//               INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in accordance with the terms of that agreement.
//        Copyright (c) 2005-2013 Intel Corporation. All Rights Reserved.
//


#ifndef __DECODE_D3D_RENDER_H__
#define __DECODE_D3D_RENDER_H__

typedef void* WindowHandle;
typedef void* Handle;
class CHWDevice;

#if defined(_WIN32) || defined(_WIN64)

struct sWindowParams
{
    LPCTSTR lpClassName;
    LPCTSTR lpWindowName;
    DWORD dwStyle;
    int nx;
    int ny;
    int ncell;
    int nAdapter;
    int nWidth;
    int nHeight;
    HWND hWndParent;
    HMENU hMenu;
    HINSTANCE hInstance;
    LPVOID lpParam;
    bool bFullScreen; // Stretch window to full screen
};

class CDecodeD3DRender
{
public:

    CDecodeD3DRender();
    virtual ~CDecodeD3DRender();

    virtual mfxStatus Init(sWindowParams pWParams);
    virtual mfxStatus RenderFrame(mfxFrameSurface1 *pSurface, mfxFrameAllocator *pmfxAlloc);
    virtual VOID UpdateTitle(double fps);

    HWND GetWindowHandle();

    VOID OnDestroy(HWND hwnd);
    VOID OnKey(HWND hwnd, UINT vk, BOOL fDown, int cRepeat, UINT flags);
    VOID ChangeWindowSize(bool bFullScreen);

    void SetHWDevice(CHWDevice *dev)
    {
        m_hwdev = dev;
    }
protected:
    void AdjustWindowRect(RECT *rect);

    CHWDevice *m_hwdev;

    sWindowParams       m_sWindowParams;
    HWND                m_Hwnd;
    RECT                m_rect;
    DWORD               m_style;

    bool EnableDwmQueuing();
    static BOOL CALLBACK MonitorEnumProc(HMONITOR ,HDC ,LPRECT lprcMonitor,LPARAM dwData);
    static bool m_bIsMonitorFound;

    bool                 m_bDwmEnabled;
    int                  m_nMonitorCurrent;
    ::RECT               m_RectWindow;
};
#endif // #if defined(_WIN32) || defined(_WIN64)

#endif // __DECODE_D3D_RENDER_H__