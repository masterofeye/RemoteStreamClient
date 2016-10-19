#include "TCPServer.h"
namespace RW{
    namespace CSR{
        DWORD TCPServer::GetCurentIP()
        {
            WSADATA wsaData;
            WSAStartup(MAKEWORD(1, 1), &wsaData);

            char HostName[1024];
            DWORD m_HostIP = 0;

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
            return m_HostIP;
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
            int err;
            char size_c[4];
            std::string addresses = GetAllIPAddresses();
            int addresses_length = addresses.length();
            memcpy(size_c, &addresses_length, sizeof(addresses_length));
            err = ::send(s1, size_c, sizeof(size_c), 0);
            err = ::send(s1, addresses.c_str(), (size_t)((unsigned)(addresses.length())), 0);
        }

        void TCPServer::ProcessSTARTSTOPCommand(SOCKET s1, std::vector<SOCKET>* sockets)
        {
            int size = (*sockets).size();
            for (int n : (*sockets)) {
                send(n, "lol", 4, 0);
            }
        }

        std::string TCPServer::ReceiveMessage(SOCKET s1)
        {
            int err = 0;
            char size_c[4] = { 0 };
            int size = 0;
            err = ::recv(s1, size_c, sizeof(size_c), 0);
            size = *((int*)size_c);
            char *string = new char[size];
            err = ::recv(s1, string, size, 0);
            char sep[1];
            char *next_token = NULL;
            sep[0] = '\0';
            strtok_s(string, sep, &next_token);
            std::cout << string << std::endl;
            std::string message(string);
            delete string;
            return message;
        }

        void TCPServer::ThreadProcess(SOCKET s1, std::vector<SOCKET>* sockets)
        {
            //QThread thread1;
            for (;;)
            {
                std::string message = ReceiveMessage(s1);

                if (message == "GETADDR")
                    ProcessGETADDRCommand(s1);
                if (message == "STOP")
                    ProcessSTARTSTOPCommand(s1, sockets);
                if (message == "DISCONNECT")
                {
                    closesocket(s1);
                    break;
                }
            }
        }

        TCPServer::TCPServer()
        {
            WORD wVersionRequested;
            WSADATA wsaData;
            wVersionRequested = MAKEWORD(2, 2);
            WSAStartup(wVersionRequested, &wsaData);

            SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

            GetCurentIP();

            struct sockaddr_in sin;
            sin.sin_family = AF_INET;
            sin.sin_port = htons(80);
            sin.sin_addr.s_addr = INADDR_ANY;

            int err = ::bind(s, (LPSOCKADDR)&sin, sizeof(sin));
            err = ::listen(s, SOMAXCONN);

            //vector<thread*> threads;
            std::vector<SOCKET> sockets;

            for (;;)
            {
                sockaddr_in from;
                int fromlen = sizeof(from);
                SOCKET s1 = accept(s, (struct sockaddr*)&from, &fromlen);
                sockets.push_back(s1);
                //threads.push_back(new thread(TCPServer::ThreadProcess, s1, &sockets));
            }
            WSACleanup();
        }

        TCPServer::~TCPServer()
        {
        }
    }
}