/*********************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2005-2014 Intel Corporation. All Rights Reserved.

**********************************************************************************/

#include "mfx_samples_config.h"

#if defined(_WIN32) || defined(_WIN64)

#pragma warning(disable : 4201)

#include "igfx_s3dcontrol.h"
#include "hw_device.h"

#include <windowsx.h>

#include "sample_defs.h"
#include "decode_render.h"
#pragma warning(disable : 4100)

bool CDecodeD3DRender::m_bIsMonitorFound = false;

LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
#ifdef WIN64
    CDecodeD3DRender* pRender = (CDecodeD3DRender*)GetWindowLongPtr(hWnd, GWLP_USERDATA);
#else
    CDecodeD3DRender* pRender = (CDecodeD3DRender*)LongToPtr(GetWindowLongPtr(hWnd, GWL_USERDATA));
#endif
    if (pRender)
    {
        switch(message)
        {
            HANDLE_MSG(hWnd, WM_DESTROY, pRender->OnDestroy);
            HANDLE_MSG(hWnd, WM_KEYUP,   pRender->OnKey);
        }
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

CDecodeD3DRender::CDecodeD3DRender()
{
    m_bDwmEnabled = false;
    m_nMonitorCurrent = 0;

    m_hwdev = NULL;
    MSDK_ZERO_MEMORY(m_sWindowParams);
    m_Hwnd = 0;
    MSDK_ZERO_MEMORY(m_rect);
    m_style = 0;
}

BOOL CALLBACK CDecodeD3DRender::MonitorEnumProc(HMONITOR /*hMonitor*/,
                                                HDC /*hdcMonitor*/,
                                                LPRECT lprcMonitor,
                                                LPARAM dwData)
{
    CDecodeD3DRender * pRender = reinterpret_cast<CDecodeD3DRender *>(dwData);
    RECT r = {0};
    if (NULL == lprcMonitor)
        lprcMonitor = &r;

    if (pRender->m_nMonitorCurrent++ == pRender->m_sWindowParams.nAdapter)
    {
        pRender->m_RectWindow = *lprcMonitor;
        m_bIsMonitorFound = true;
    }
    return TRUE;
}

CDecodeD3DRender::~CDecodeD3DRender()
{
    if (m_Hwnd)
        DestroyWindow(m_Hwnd);

    //DestroyTimer();
}

mfxStatus CDecodeD3DRender::Init(sWindowParams pWParams)
{
    mfxStatus sts = MFX_ERR_NONE;

    // window part
    m_sWindowParams = pWParams;

    WNDCLASS window;
    MSDK_ZERO_MEMORY(window);

    window.lpfnWndProc= (WNDPROC)WindowProc;
    window.hInstance= GetModuleHandle(NULL);;
    window.hCursor= LoadCursor(NULL, IDC_ARROW);
    window.lpszClassName= m_sWindowParams.lpClassName;

    if (!RegisterClass(&window))
        return MFX_ERR_UNKNOWN;

    EnumDisplayMonitors(NULL, NULL, &CDecodeD3DRender::MonitorEnumProc, (LPARAM)this);
    if(!m_bIsMonitorFound)
        return MFX_ERR_NOT_FOUND;

    ::RECT displayRegion = {CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT};

    //right and bottom fields consist of width and height values of displayed reqion
    if (0 != m_sWindowParams.nx )
    {
        displayRegion.right  = (m_RectWindow.right - m_RectWindow.left)  / m_sWindowParams.nx;
        displayRegion.bottom = (m_RectWindow.bottom - m_RectWindow.top)  / m_sWindowParams.ny;
        displayRegion.left   = displayRegion.right * (m_sWindowParams.ncell % m_sWindowParams.nx) + m_RectWindow.left;
        displayRegion.top    = displayRegion.bottom * (m_sWindowParams.ncell / m_sWindowParams.nx) + m_RectWindow.top;
    }
    else
    {
        displayRegion.right = pWParams.nWidth;
        displayRegion.bottom = pWParams.nHeight;
        displayRegion.left = m_RectWindow.left;
        displayRegion.top= m_RectWindow.top;
    }

    //no title window style if required
    DWORD dwStyle = NULL == m_sWindowParams.lpWindowName ?  WS_POPUP|WS_BORDER|WS_MAXIMIZE : WS_OVERLAPPEDWINDOW;

    m_Hwnd = CreateWindowEx(NULL,
        m_sWindowParams.lpClassName,
        m_sWindowParams.lpWindowName,
        !m_sWindowParams.bFullScreen ? dwStyle : (WS_POPUP),
        !m_sWindowParams.bFullScreen ? displayRegion.left : 0,
        !m_sWindowParams.bFullScreen ? displayRegion.top : 0,
        !m_sWindowParams.bFullScreen ? displayRegion.right : GetSystemMetrics(SM_CXSCREEN),
        !m_sWindowParams.bFullScreen ? displayRegion.bottom : GetSystemMetrics(SM_CYSCREEN),
        m_sWindowParams.hWndParent,
        m_sWindowParams.hMenu,
        m_sWindowParams.hInstance,
        m_sWindowParams.lpParam);

    if (!m_Hwnd)
        return MFX_ERR_UNKNOWN;

    ShowWindow(m_Hwnd, SW_SHOWDEFAULT);
    UpdateWindow(m_Hwnd);

#ifdef WIN64
    SetWindowLongPtr(m_Hwnd, GWLP_USERDATA, (LONG_PTR)this);
#else
    SetWindowLong(m_Hwnd, GWL_USERDATA, PtrToLong(this));
#endif

    m_hwdev->SetHandle((mfxHandleType)MFX_HANDLE_DEVICEWINDOW, m_Hwnd);
    sts = m_hwdev->Reset();
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    return sts;
}

mfxStatus CDecodeD3DRender::RenderFrame(mfxFrameSurface1 *pSurface, mfxFrameAllocator *pmfxAlloc)
{
    RECT rect;
    GetClientRect(m_Hwnd, &rect);
    if (IsRectEmpty(&rect))
        return MFX_ERR_UNKNOWN;

    mfxStatus sts = m_hwdev->RenderFrame(pSurface, pmfxAlloc);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    return sts;
}

HWND CDecodeD3DRender::GetWindowHandle()
{
    if (!m_Hwnd)
    {
        EnumDisplayMonitors(NULL, NULL, &CDecodeD3DRender::MonitorEnumProc, (LPARAM)this);
        POINT point  = {m_RectWindow.left, m_RectWindow.top};
        m_Hwnd = WindowFromPoint(point);
        m_nMonitorCurrent = 0;
        m_bIsMonitorFound = false;
    }
    return m_Hwnd;
}

VOID CDecodeD3DRender::UpdateTitle(double fps)
{
    if (m_Hwnd)
    {
        MSG msg;
        MSDK_ZERO_MEMORY(msg);
        while (msg.message != WM_QUIT && PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (NULL != m_sWindowParams.lpWindowName) {
            TCHAR str[20];
            _stprintf_s(str, 20, _T("fps=%.2lf"), fps );

            SetWindowText(m_Hwnd, str);
        }
    }
}

VOID CDecodeD3DRender::OnDestroy(HWND /*hwnd*/)
{
    PostQuitMessage(0);
}

VOID CDecodeD3DRender::OnKey(HWND hwnd, UINT vk, BOOL fDown, int cRepeat, UINT flags)
{
    if (TRUE == fDown)
        return;

    if ('1' == vk && false == m_sWindowParams.bFullScreen)
        ChangeWindowSize(true);
    else if (true == m_sWindowParams.bFullScreen)
        ChangeWindowSize(false);
}

void CDecodeD3DRender::AdjustWindowRect(RECT *rect)
{
    int cxmax = GetSystemMetrics(SM_CXMAXIMIZED);
    int cymax = GetSystemMetrics(SM_CYMAXIMIZED);
    int cxmin = GetSystemMetrics(SM_CXMINTRACK);
    int cymin = GetSystemMetrics(SM_CYMINTRACK);
    int leftmax = cxmax - cxmin;
    int topmax = cymax - cxmin;
    if (rect->left < 0)
        rect->left = 0;
    if (rect->left > leftmax)
        rect->left = leftmax;
    if (rect->top < 0)
        rect->top = 0;
    if (rect->top > topmax)
        rect->top = topmax;

    if (rect->right < rect->left + cxmin)
        rect->right = rect->left + cxmin;
    if (rect->right - rect->left > cxmax)
        rect->right = rect->left + cxmax;

    if (rect->bottom < rect->top + cymin)
        rect->bottom = rect->top + cymin;
    if (rect->bottom - rect->top > cymax)
        rect->bottom = rect->top + cymax;
}

VOID CDecodeD3DRender::ChangeWindowSize(bool bFullScreen)
{
    HMONITOR hMonitor = MonitorFromWindow(m_Hwnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFOEX mi;
    mi.cbSize = sizeof(mi);
    GetMonitorInfo(hMonitor, &mi);

    WINDOWINFO wndInfo;
    wndInfo.cbSize = sizeof(WINDOWINFO);
    GetWindowInfo(m_Hwnd, &wndInfo);

    if(!m_sWindowParams.bFullScreen)
    {
        m_rect = wndInfo.rcWindow;
        m_style = wndInfo.dwStyle;
    }

    m_sWindowParams.bFullScreen = bFullScreen;

    if(!bFullScreen)
    {
        AdjustWindowRectEx(&m_rect,0,0,0);
        SetWindowLong(m_Hwnd, GWL_STYLE, m_style);
        SetWindowPos(m_Hwnd, HWND_NOTOPMOST,
            m_rect.left , m_rect.top ,
            abs(m_rect.right - m_rect.left), abs(m_rect.bottom - m_rect.top),
            SWP_SHOWWINDOW);
    }
    else
    {
        SetWindowLong(m_Hwnd, GWL_STYLE, WS_POPUP);
        SetWindowPos(m_Hwnd, HWND_NOTOPMOST,mi.rcMonitor.left , mi.rcMonitor.top,
            abs(mi.rcMonitor.left - mi.rcMonitor.right), abs(mi.rcMonitor.top - mi.rcMonitor.bottom), SWP_SHOWWINDOW);
    }
}

#endif // #if defined(_WIN32) || defined(_WIN64)
