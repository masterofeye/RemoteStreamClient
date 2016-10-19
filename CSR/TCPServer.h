
#pragma once
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _WINSOCKAPI_ 

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <process.h>
#include <iostream>
#include <QtCore/qobject.h>
//#include <QtCore/qthread.h>

#include "Config.h"

#pragma comment(lib, "ws2_32.lib")
#define PORT 3135
#define IP "127.0.0.1"

namespace RW{
    namespace CSR{
        class TCPServer 
        {
        public:
            DWORD GetCurentIP();
            std::string GetAllIPAddresses();
            void ProcessGETADDRCommand(SOCKET s1);
            void ProcessSTARTSTOPCommand(SOCKET s1, std::vector<SOCKET>* sockets);
            void ThreadProcess(SOCKET s1, std::vector<SOCKET>* sockets);
            void StartInSeparateThread();

            TCPServer();
            ~TCPServer();

        private:
            SOCKET s;
            sockaddr_in serv_addr;

            std::string ReceiveMessage(SOCKET s1);
            void SendData(SOCKET s1, void* pData);

        };
    }
}