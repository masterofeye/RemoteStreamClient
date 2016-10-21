#include "TCPServer.h"
namespace RW{
    namespace CSR{
        DWORD TCPServer::GetCurentIP()
        {
            DWORD dwHostIP = 0;

            WSADATA wWsaData;
            if (WSAStartup(MAKEWORD(1, 1), &wWsaData) != 0){
                printf("TCPServer::GetCurentIP: WSACleanup failed");
                return dwHostIP;
            }
            char HostName[1024];

            if (::gethostname(HostName, 1024) == 0)
            {
                if (LPHOSTENT lphost = ::gethostbyname(HostName))
                {
                    int i = 0;
                    struct in_addr addr;
                    while (lphost->h_addr_list[i] != 0)
                    {
                        addr.s_addr = *(u_long *)lphost->h_addr_list[i++];
                        std::cout << "IPv4 Address:" << inet_ntoa(addr) << std::endl;
                    }
                }
            }
            if (WSACleanup() != 0){
                printf("TCPServer::GetCurentIP: WSACleanup failed");
            }

            // TODO: Where do you assign a value to m_HostIP?
            return dwHostIP;
        }

        std::string TCPServer::GetAllIPAddresses()
        {
            char* addrs[3] = { "127.0.0.1", "127.0.0.2", "127.0.0.3" };
            std::string result = "";
            for (int i = 0; i < 3; i++)
            {
                result += addrs[i];
                result += '\n';
            }
            result.erase(result.length() - 1, 1);
            result += '\0';
            return result;
        }

        void TCPServer::ProcessGETADDRCommand(SOCKET s1)
        {
            int iErr;
            char cSize[4];
            std::string strAddresses = GetAllIPAddresses();
            size_t sAdressesLength = strAddresses.length();
            memcpy(cSize, &sAdressesLength, sizeof(sAdressesLength));
            if (cSize == NULL){
                printf("TCPServer::ProcessGETADDRCommand: memcpy failed");
                return;
            }

            iErr = ::send(s1, cSize, sizeof(cSize), 0);
            if (iErr != 0){
                printf("TCPServer::ProcessGETADDRCommand: send cSize failed");
                return;
            }
            iErr = ::send(s1, strAddresses.c_str(), (size_t)((unsigned)(strAddresses.length())), 0);
            if (iErr != 0){
                printf("TCPServer::ProcessGETADDRCommand: send strAddresses failed");
                return;
            }
        }

        void TCPServer::ProcessSTARTSTOPCommand(SOCKET s1, std::vector<SOCKET>* sockets)
        {
            size_t sSize = (*sockets).size();
            for (int iIndex : (*sockets)) {
                int iErr = ::send(iIndex, "lol", 4, 0);
                if (iErr != 0){
                    printf("TCPServer::ProcessSTARTSTOPCommand: send failed");
                    return;
                }
            }
            return;
        }

        std::string TCPServer::ReceiveMessage(SOCKET s1)
        {
            int iErr = 0;
            char cSize[4] = { 0 };
            int iSize = 0;
            iErr = ::recv(s1, cSize, sizeof(cSize), 0);
            if (iErr != 0){
                printf("TCPServer::ReceiveMessage: recv cSize failed");
                return "";
            }
            iSize = *((int*)cSize);
            char *pcMsg = new char[iSize];
            iErr = ::recv(s1, pcMsg, iSize, 0);
            if (iErr != 0){
                printf("TCPServer::ReceiveMessage: recv pcMsg failed");
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

        void TCPServer::ThreadProcess(SOCKET s1, std::vector<SOCKET>* sockets)
        {
            //QThread thread1;
            for (;;)
            {
                std::string strMessage = ReceiveMessage(s1);

                if (strMessage == "GETADDR")
                    ProcessGETADDRCommand(s1);
                if (strMessage == "STOP")
                    ProcessSTARTSTOPCommand(s1, sockets);
                if (strMessage == "DISCONNECT")
                {
                    if (::closesocket(s1) != 0){
                        printf("TCPServer::ThreadProcess: closesocket failed");
                        return;
                    }
                    break;
                }
            }
        }

        TCPServer::TCPServer()
        {
            WORD wVersionRequested;
            WSADATA wsaData;
            wVersionRequested = MAKEWORD(2, 2);
            if (WSAStartup(wVersionRequested, &wsaData) != 0){
                printf("TCPServer::TCPServer: WSAStartup failed");
                return;
            }

            SOCKET sock1 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

            GetCurentIP();

            struct sockaddr_in stSockAddrIn;
            stSockAddrIn.sin_family = AF_INET;
            stSockAddrIn.sin_port = htons(80);
            stSockAddrIn.sin_addr.s_addr = INADDR_ANY;

            int iErr = ::bind(sock1, (LPSOCKADDR)&stSockAddrIn, sizeof(stSockAddrIn));
            if (iErr != 0){
                printf("TCPServer::TCPServer: bind failed");
                return;
            }
            iErr = ::listen(sock1, SOMAXCONN);
            if (iErr != 0){
                printf("TCPServer::TCPServer: listen failed");
                return;
            }

            //vector<thread*> threads;
            std::vector<SOCKET> vecSockets;

            for (;;)
            {
                sockaddr_in stSockAddrFrom;
                int iLengthFrom = sizeof(stSockAddrFrom);
                SOCKET s1 = ::accept(sock1, (struct sockaddr*)&stSockAddrFrom, &iLengthFrom);
                vecSockets.push_back(s1);
                //threads.push_back(new thread(TCPServer::ThreadProcess, s1, &sockets));
            }
            if (WSACleanup() != 0){
                printf("TCPServer::TCPServer: WSACleanup failed");
                return;
            }
        }

        TCPServer::~TCPServer()
        {
        }
    }
}