#include "TCPClient.h"

namespace RW{
    namespace CCL{
        DWORD TCPClient::GetCurentIP()
        {
            WSADATA wsaData;
            WSAStartup(MAKEWORD(1, 1), &wsaData);

            char HostName[1024];
            DWORD dwHostIP = 0;

            if (!gethostname(HostName, 1024))
            {
                if (LPHOSTENT lphost = gethostbyname(HostName))
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
            WSACleanup();
            return dwHostIP;
        }

        SOCKET TCPClient::Connect(char *pcHost, bool bIsOperator)
        {
            in_addr ip = *(GetIPbyHostname(pcHost));

            struct sockaddr_in anAddr;
            anAddr.sin_family = AF_INET;
            anAddr.sin_port = htons(80);
            anAddr.sin_addr.S_un.S_addr = ip.S_un.S_addr;//inet_addr(IP); //"192.168.1.145");
            SOCKET s1 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            connect(s1, (struct sockaddr *)&anAddr, sizeof(struct sockaddr));
            return s1;
        }

        void TCPClient::SendCommand(SOCKET s1, std::string command)
        {
            command += '\0';
            char size_c[4];
            int size = command.size();
            memcpy(size_c, &size, sizeof(size));
            send(s1, size_c, sizeof(size_c), 0);
            send(s1, command.c_str(), (size_t)((unsigned)(command.length())), 0);
        }

        void TCPClient::ProcessGETADDRCommand(SOCKET s1)
        {
            SendCommand(s1, std::string("GETADDR"));

            int rc;
            unsigned size = 0;
            char size_c[4];

            int addresses_length = 0;
            rc = recv(s1, size_c, sizeof(size_c), 0);
            size = *((int*)size_c);
            char *addresses = new char[size];
            rc = recv(s1, addresses, size, 0);
            char sep[2];
            char *next_token = NULL;
            sep[0] = '\n';
            strtok_s(addresses, sep, &next_token);
            std::cout << addresses << std::endl;
            while (next_token[0] != '\0')
            {
                addresses = next_token;
                strtok_s(addresses, sep, &next_token);
                std::cout << addresses << std::endl;
            }
        }

        void TCPClient::ProcessDISCONNECTCommand(SOCKET s1)
        {
            SendCommand(s1, std::string("DISCONNECT"));
            closesocket(s1);
        }

        TCPClient::TCPClient()
        {
            WORD wVersionRequested;
            WSADATA wsaData;
            wVersionRequested = MAKEWORD(2, 2);
            WSAStartup(wVersionRequested, &wsaData);

            int rc;

            GetCurentIP();

            char isOperator;
            std::cout << "Are you operator?" << std::endl;
            isOperator = getchar();

            SOCKET s1;// = Connect();

            ProcessGETADDRCommand(s1);

            if (isOperator != 'y')
            {
                std::cout << "Waiting for STOP command" << std::endl;
                char buf[4];
                recv(s1, buf, 5, 0);
                std::cout << buf << std::endl;
            }
            else
            {
                system("pause");
                SendCommand(s1, std::string("STOP"));
                std::cout << "Waiting for STOP command" << std::endl;
                char buf[4];
                recv(s1, buf, 5, 0);
                std::cout << buf << std::endl;
            }

            ProcessDISCONNECTCommand(s1);
            WSACleanup();
        }


        TCPClient::~TCPClient()
        {
        }

        in_addr *TCPClient::GetIPbyHostname(char* pcHost){
            if (pcHost == NULL){
                printf("No Hostname %s\n", pcHost);
                return NULL;
            }
            hostent * record = gethostbyname(pcHost);
            if (record == NULL)
            {
                printf("%s is unavailable\n", pcHost);
                return NULL;
            }
            in_addr *address = (in_addr *)record->h_addr;
            //string ip_address = inet_ntoa(*address);
            return address;
        }
    }
}