#include "TCPClient.hpp"
#include <qthread.h>

namespace RW{
    namespace CCL{
        std::string TCPClient::GetCurentIP()
        {
            WSADATA wsaData;
            WSAStartup(MAKEWORD(1, 1), &wsaData);

            char* pcCurrentIP;

            char cHostName[1024];

            if (::gethostname(cHostName, 1024) == 0)
            {
                if (LPHOSTENT stLPHOST = ::gethostbyname(cHostName))
                {
                    int iIndex = 0;
                    struct in_addr addr;
                    while (stLPHOST->h_addr_list[iIndex] != 0)
                    {
                        addr.s_addr = *(u_long *)stLPHOST->h_addr_list[iIndex];
                        //std::cout << "IPv4 Address:" << inet_ntoa(addr) << std::endl;
                        if (iIndex == 0)
                            pcCurrentIP = ::inet_ntoa(addr);
                        iIndex++;
                    }
                }
            }
            WSACleanup();
            return std::string(pcCurrentIP);
        }

        SOCKET TCPClient::ConnectToServer(char *pcHost, char cIP[]) //, bool bIsOperator)
        {
            struct sockaddr_in stAddr;
            stAddr.sin_family = AF_INET;
            stAddr.sin_port = ::htons(80);
            stAddr.sin_addr.S_un.S_addr = ::inet_addr(cIP); //"192.168.1.145");
            SOCKET s1 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            ::connect(s1, (struct sockaddr *)&stAddr, sizeof(struct sockaddr));
            return s1;
        }

        void TCPClient::SendCommand(SOCKET s1, std::string strCommand)
        {
            strCommand += '\0';
            char cSize[4];
            int iSize = strCommand.size();
            memcpy(cSize, &iSize, sizeof(iSize));
            if (cSize == NULL){
                printf("TCPClient::SendCommand: memcpy failed ");
                return;
            }

            send(s1, cSize, sizeof(cSize), 0);
            send(s1, cSize, sizeof(cSize), 0);
            send(s1, strCommand.c_str(), (size_t)((unsigned)(strCommand.length())), 0);
        }

        void TCPClient::ProcessGETADDRCommand(SOCKET s1)
        {
            SendCommand(s1, std::string("GETADDR"));

            int iRetVal;
            unsigned int uSize = 0;
            char cSize[4];

            int iAddressLength = 0;
            iRetVal = ::recv(s1, cSize, sizeof(cSize), 0);
            if (iRetVal != 0){
                printf("TCPClient::ProcessGETADDRCommand: recv failed for socket %d!", s1);
                return;
            }
            uSize = *((int*)cSize);
            char *pcAddresses = new char[uSize];
            iRetVal = ::recv(s1, pcAddresses, uSize, 0);
            if (iRetVal != 0){
                printf("TCPClient::ProcessGETADDRCommand: recv failed for socket %d!", s1);
                return;
            }
            char cSep[2];
            char *pcNextToken = NULL;
            cSep[0] = '\n';
            if (::strtok_s(pcAddresses, cSep, &pcNextToken) != NULL)
            {
                std::cout << pcAddresses << std::endl;
                while (pcNextToken[0] != '\0')
                {
                    pcAddresses = pcNextToken;
                    if (strtok_s(pcAddresses, cSep, &pcNextToken) != NULL){
                        std::cout << pcAddresses << std::endl;
                    }
                }
            }
        }

        void TCPClient::ProcessDISCONNECTCommand(SOCKET s1)
        {
            SendCommand(s1, std::string("DISCONNECT"));
            if (::closesocket(s1) != 0){
                printf("TCPClient::ProcessDISCONNECTCommand: closesocket failed for socket %d!", s1);
                return;
            }
        }

        void TCPClient::ProcessCONNECTCommand(SOCKET s1, bool bIsOperator)
        {
            SendCommand(s1, std::string("CONNECT"));
            SendCommand(s1, GetCurentIP());
            SendCommand(s1, std::string("true")); //todo: change second parameter
            std::string strMessage = ReceiveMessage(s1);
            //send to pipeline
        }

        void TCPClient::ProcessREMOVECommand(SOCKET s1, char cIdToRemove[])
        {
            SendCommand(s1, std::string("REMOVE"));
            SendCommand(s1, std::string(cIdToRemove));
        }

        std::list<char*>  TCPClient::ProcessGETCLIENTSCommand(SOCKET s1)
        {
            SendCommand(s1, std::string("GETCLIENTS"));
            std::string strMessage = ReceiveMessage(s1);

            std::list<char*> plstAddresses;

            char* pcAddresses = new char[strMessage.size()];//as 1 char space for null is also required
            ::strcpy(pcAddresses, strMessage.c_str());
            char cSep[2];
            char *pcNextToken = NULL;
            cSep[0] = '\n';
            ::strtok_s(pcAddresses, cSep, &pcNextToken);
            //cout << addresses << endl;
            plstAddresses.push_back(pcAddresses);
            while (pcNextToken[0] != '\0')
            {
                pcAddresses = pcNextToken;
                if (::strtok_s(pcAddresses, cSep, &pcNextToken) != NULL){
                    //std::cout << addresses << std::endl;
                    plstAddresses.push_back(pcAddresses);
                }
            }
            return plstAddresses;
        }

        std::string TCPClient::ReceiveMessage(SOCKET s1)
        {
            int iErr = 0;
            char cSize[4] = { 0 };
            int iSize = 0;
            iErr = ::recv(s1, cSize, sizeof(cSize), 0);
            if (iErr != 0){
                printf("TCPClient::ReceiveMessage: recv failed for socket %d!", s1);
                return "";
            }
            iSize = *((int*)cSize);
            char *pcMsg = new char[iSize];
            iErr = ::recv(s1, pcMsg, iSize, 0);
            if (iErr != 0){
                printf("TCPClient::ReceiveMessage: recv failed for socket %d!", s1);
                return "";
            }
            char cSep[1];
            char *pcNextToken = NULL;
            cSep[0] = '\0';
            std::cout << pcMsg << std::endl;
            std::string strMessage(pcMsg);
            delete pcMsg;
            return strMessage;
        }

        void TCPClient::SetConfig(sSimpleConfig stConfiguration)
        {
            //char cMsg[sizeof(sSimple)];
            //memcpy(cMsg, (void*)dummy, sizeof(sSimple));
            SendCommand(m_sockClient, std::string("SENDCONFIG"));
            //char* config = 
            //SendCommand(client_sock, )
        }

        int TCPClient::Disconnect()
        {
            SendCommand(m_sockClient, std::string("DISCONNECT"));
            if (::closesocket(m_sockClient) != 0){
                printf("ProcessDISCONNECTCommand: closesocket failed for socket %d!", m_sockClient);
                return -1;
            }
            return 0;
        }

        int TCPClient::ConnectToSession(char IP[], bool bIsOperator)
        {
            m_sockClient = ConnectToServer("12345", IP);
            m_sockAlert = ConnectToServer("12345", IP);
            ProcessCONNECTCommand(m_sockClient, bIsOperator);
            TCPClientWrapper* pWrapper = new TCPClientWrapper(m_sockAlert);
            QThread* pThread = new QThread;
            pWrapper->moveToThread(pThread);
            QObject::connect(pThread, SIGNAL(started()()), pWrapper, SLOT(Process));
            pThread->start();
            return 0;
        }

        int TCPClient::RemoveClient(char cClientID[])//now ip used as id
        {
            ProcessREMOVECommand(m_sockClient, cClientID);
            return 0;
        }

        std::list<char*> TCPClient::GetClientList()
        {
            std::list<char*> lstClients;
            // TODO: Implement
            return lstClients;
        }

        void TCPClient::Process()
        {
            //WORD wVersionRequested;
            //WSADATA wsaData;
            //wVersionRequested = MAKEWORD(2, 2);
            //WSAStartup(wVersionRequested, &wsaData);
            //
            //int rc;
            //
            //char isOperator;
            ////cout << "Are you operator?" << endl;
            ////isOperator = getchar();
            //
            ////SOCKET s1 = ConnectToServer("12345");
            //
            //TCPClientWrapper* wrapper = new TCPClientWrapper(s1);
            //QThread* thread = new QThread;
            //wrapper->moveToThread(thread);
            //
            //ProcessCONNECTCommand(s1);
            //
            //ProcessGETADDRCommand(s1);
            //
            ////if (isOperator != 'y')
            ////{
            ////	cout << "Waiting for STOP command" << endl;
            ////	char buf[4];
            ////	recv(s1, buf, 5, 0);
            ////	cout << buf << endl;
            ////}
            ////else
            ////{
            ////	system("pause");
            ////	SendCommand(s1, string("STOP"));
            ////	cout << "Waiting for STOP command" << endl;
            ////	char buf[4];
            ////	recv(s1, buf, 5, 0);
            ////	cout << buf << endl;
            ////}
            //
            //ProcessDISCONNECTCommand(s1);
            //WSACleanup();
        }

        TCPClient::TCPClient()
        {
        }

        TCPClient::~TCPClient()
        {
        }

        TCPClientWrapper::TCPClientWrapper(SOCKET s1)
        {
            m_sS1 = s1;
        }

        TCPClientWrapper::~TCPClientWrapper()
        {
        }

        std::string TCPClientWrapper::ReceiveMessage(SOCKET s1)
        {
            int iErr = 0;
            char cSize[4] = { 0 };
            int iSize = 0;
            iErr = ::recv(s1, cSize, sizeof(cSize), 0);
            if (iErr != 0){
                printf("TCPClientWrapper::ReceiveMessage: recv failed for socket %d!", s1);
                return "";
            }
            iSize = *((int*)cSize);
            char *pcMsg = new char[iSize];
            iErr = ::recv(s1, pcMsg, iSize, 0);
            if (iErr != 0){
                printf("TCPClientWrapper::ReceiveMessage: recv failed for socket %d!", s1);
                return "";
            }
            char cSep[1];
            char *pcNextToken = NULL;
            cSep[0] = '\0';
            if (::strtok_s(pcMsg, cSep, &pcNextToken) != NULL){
                std::cout << pcMsg << std::endl;
                std::string strMessage(pcMsg);
                delete pcMsg;
                return strMessage;
            }
            else{
                delete pcMsg;
                return "";
            }
        }

        void TCPClientWrapper::ProcessSTOPCommand(SOCKET s1)
        {
            emit Stop();
        }

        void TCPClientWrapper::ProcessSENDCONFIGCommand(SOCKET s1)
        {
            std::string config = ReceiveMessage(s1);
        }

        void TCPClientWrapper::Process()
        {
            for (;;)
            {
                std::string message = ReceiveMessage(m_sS1);

                if (message == "STOP")
                    ProcessSTOPCommand(m_sS1);
                if (message == "SENDCONFIG")
                    ProcessSENDCONFIGCommand(m_sS1);
            }
        }
    }
}