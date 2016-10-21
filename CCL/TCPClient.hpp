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
#include "..\RWPluginInterface\Config.h"

#pragma comment(lib, "ws2_32.lib")
#define PORT 3135

namespace RW{

    namespace CCL{

        class TCPClient : public QObject
        {
            Q_OBJECT

        private:
            in_addr *GetIPbyHostname(char* pcHost);
            std::string GetCurentIP();//OK

            void SendCommand(SOCKET s1, std::string command);//OK
            void ProcessCONNECTCommand(SOCKET s1, bool isOperator);//OK
            void ProcessREMOVECommand(SOCKET s1, char idToRemove[]);//OK
            std::list<char*> ProcessGETCLIENTSCommand(SOCKET s1);//OK
            std::string ReceiveMessage(SOCKET s1);//OK

            sSimpleConfig GetConfig();	// After Connect Requesting Server 
            char* GetUDPAddresses();
            int ViewMsg();

        private:
            sSimpleConfig m_stDummy;
            SOCKET m_sockClient;
            SOCKET m_sockAlert;
            char m_cBuf[1024];

            bool     m_bIsOperator;
            sSimpleConfig *m_pConfiguration;
            std::list<char*> m_lstAddresses;

        private:
            //**********************************************************//
            //****************** Connection to Pipeline ****************//
            //**********************************************************//
            //void SetConfigForPipeline(sSimpleConfig stConfiguration);
            void SetUDPAdresses(std::list<char*> lstAddresses);
            int RunPipeline();
            int StopPipeline();

        public:
            SOCKET ConnectToServer(char *pcHost, char IP[]);//, bool bIsOperator = false);
            void ProcessGETADDRCommand(SOCKET s1);//OK
            void ProcessDISCONNECTCommand(SOCKET s1);//OK

            TCPClient();
            ~TCPClient();

        public slots:
            //**********************************************************//
            //***************** Connection to Server *******************//
            //**********************************************************//
            int ConnectToSession(char IP[], bool bIsOperator = false);//OK
            int Disconnect();//OK
            int RemoveClient(char ClientID[]); // Operator only //OK
            std::list<char*> GetClientList(); // Operator only
            void SetConfig(sSimpleConfig stConfiguration); // Operator only, Called by GUI, Sending to Server

            void Process();

        signals:
            void SetConfigForPipeline(sSimpleConfig stConfiguration);
            void started();
        };

        class TCPClientWrapper : public QObject
        {
            Q_OBJECT

        public:
            void ProcessSENDCONFIGCommand(SOCKET s1);
            void ProcessSTOPCommand(SOCKET s1);
            std::string ReceiveMessage(SOCKET s1);

            SOCKET m_sS1;

            TCPClientWrapper(SOCKET s1);
            ~TCPClientWrapper();

        public slots:
            void Process();
        signals:
            void Stop();
        };
    }
}