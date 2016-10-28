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

		struct ClientInfo
		{
			std::string m_strIP;
			bool m_boIsApproved;
		};

        class TCPClient : public QObject
        {
            Q_OBJECT

		public:
			SOCKET oConnectToServer(char *pcHost, char IP[]);
			void vProcessGETADDRCommand(SOCKET s1);
			void vProcessDISCONNECTCommand(SOCKET s1);

			TCPClient();
			~TCPClient();

        private:
            in_addr *GetIPbyHostname(char* pcHost);
            std::string GetCurentIP();

            void vSendCommand(SOCKET s1, std::string command);
            void vProcessCONNECTCommand(SOCKET s1, bool isOperator);
            void vProcessREMOVECommand(SOCKET s1, char idToRemove[]);
            std::list<ClientInfo*> oProcessGETCLIENTSCommand(SOCKET s1);
			void vProcessSTARTCommand(SOCKET s1);
            std::string strReceiveMessage(SOCKET s1);

            sSimpleConfig oGetConfig();	// After Connect Requesting Server 
            char* cpGetUDPAddresses();
            int iViewMsg();

            SOCKET m_sockClient;
            SOCKET m_sockAlert;
            char m_cBuf[1024];

            bool m_bIsOperator;
            sSimpleConfig *m_pConfiguration;
            std::list<char*> m_lstAddresses;

        public slots:
            //**********************************************************//
            //***************** Connection to Server *******************//
            //**********************************************************//
            int iConnectToSession(char IP[], bool bIsOperator = false);
            int iDisconnect();
            int iRemoveClient(char ClientID[]); // Operator only
            std::list<ClientInfo*> oGetClientList(); // Operator only
            void vSetConfig(sSimpleConfig stConfiguration); // Operator only
			void vApprove(char ipToApprove[]); // Operator only
			int iStop();
			void vStart();

			void vStopReceived();

        signals:
            void SetConfigForPipeline(sSimpleConfig stConfiguration);
			void SetUDPAdresses(std::list<char*> lstAddresses);
			int RunPipeline();
			int StopPipeline();
            void started();
        };

        class TCPClientWrapper : public QObject
        {
            Q_OBJECT

        public:
            void vProcessSENDCONFIGCommand(SOCKET s1);
            void vProcessSTOPCommand(SOCKET s1);
            std::string strReceiveMessage(SOCKET s1);

            SOCKET m_oS1;

            TCPClientWrapper(SOCKET s1);
            ~TCPClientWrapper();

		private:
			std::list<char*> oParseMessageToList(std::string message);

        public slots:
            void Process();
        signals:
            void Stop();
			void SetConfigForPipeline(sSimpleConfig stConfiguration);
        };

		
    }
}