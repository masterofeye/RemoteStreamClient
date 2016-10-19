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
#include <thread>
#include <vector>

#include <QtCore/qobject.h>
#include "Config.h"

#pragma comment(lib, "ws2_32.lib")
#define PORT 3135

namespace RW{

    namespace CCL{
        class TCPClient
        {
        public:
            SOCKET Connect(char *pcHost, bool bIsOperator = false);
            void ProcessGETADDRCommand(SOCKET s1);
            void ProcessDISCONNECTCommand(SOCKET s1);

            TCPClient();
            ~TCPClient();

        private:
            in_addr *GetIPbyHostname(char* pcHost);
            DWORD GetCurentIP();

            void SendCommand(SOCKET s1, std::string command);
            std::string ReceiveCommand(SOCKET s1);

            SOCKET client_sock;
            char buf[1024];

        private:
            bool     m_bIsOperator;
            cConfig *m_pConfiguration;
            std::list<char*> m_lstAddresses;

        public:

            //**********************************************************//
            //***************** Connection to Server *******************//
            //**********************************************************//
            int ConnectToSession(char IP[], bool bIsOperator = false);
            int Disconnect();
            void SetConfig(cConfig stConfiguration);
            int RemoveClient(char ClientID[]);
            std::list<char*> GetClientList();


        private:
            cConfig GetConfig();
            char* GetAddresses();
            int ViewMsg();


            //**********************************************************//
            //****************** Connection to Pipeline ****************//
            //**********************************************************//
            void SetConfigForPipeline(cConfig stConfiguration);
            void SetAdresses(std::list<char*> lstAddresses);
            int RunPipeline();
            int StopPipeline();
        };
    }
}