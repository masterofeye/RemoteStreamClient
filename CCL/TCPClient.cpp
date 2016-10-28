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

		SOCKET TCPClient::oConnectToServer(char *pcHost, char cIP[]) //, bool bIsOperator)
		{
			struct sockaddr_in stAddr;
			stAddr.sin_family = AF_INET;
			stAddr.sin_port = ::htons(80);
			stAddr.sin_addr.S_un.S_addr = ::inet_addr(cIP); //"192.168.1.145");
			SOCKET s1 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			::connect(s1, (struct sockaddr *)&stAddr, sizeof(struct sockaddr));
			return s1;
		}

		void TCPClient::vSendCommand(SOCKET s1, std::string strCommand)
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

		void TCPClient::vProcessGETADDRCommand(SOCKET s1)
		{
			vSendCommand(s1, std::string("GETADDR"));

			int iRetVal;
			unsigned int uSize = 0;
			char cSize[4];

			std::list<char*> plstAddresses;

			int iAddressLength = 0;
			iRetVal = ::recv(s1, cSize, sizeof(cSize), 0);
			if (iRetVal <= 0)
			{
				printf("TCPClient::ProcessGETADDRCommand: recv failed for socket %d!", s1);
			}
			uSize = *((int*)cSize);
			char *pcAddresses = new char[uSize];
			iRetVal = ::recv(s1, pcAddresses, uSize, 0);
			if (iRetVal <= 0)
			{
				printf("TCPClient::ProcessGETADDRCommand: recv failed for socket %d!", s1);
			}
			char cSep[2];
			char *pcNextToken = NULL;
			cSep[0] = '\n';
			do
			{
				strtok_s(pcAddresses, cSep, &pcNextToken);
				plstAddresses.push_back(pcAddresses);
				pcAddresses = pcNextToken;
			} while (pcNextToken[0] != '\0');

			emit SetUDPAdresses(plstAddresses);
		}

		void TCPClient::vProcessDISCONNECTCommand(SOCKET s1)
		{
			vSendCommand(s1, std::string("DISCONNECT"));
			if (::closesocket(s1) != 0){
				printf("TCPClient::ProcessDISCONNECTCommand: closesocket failed for socket %d!", s1);
				return;
			}
		}

		void TCPClient::vProcessCONNECTCommand(SOCKET s1, bool bIsOperator)
		{
			vSendCommand(s1, std::string("CONNECT"));
			vSendCommand(s1, GetCurentIP());
			if (bIsOperator)
				vSendCommand(s1, std::string("true"));
			else
				vSendCommand(s1, std::string("false"));
			std::string strMessage = strReceiveMessage(s1);
			//send to pipeline
		}

		void TCPClient::vProcessREMOVECommand(SOCKET s1, char cIdToRemove[]) // now ip used as id
		{
			vSendCommand(s1, std::string("REMOVE"));
			vSendCommand(s1, std::string(cIdToRemove));
		}

		std::list<ClientInfo*> TCPClient::oProcessGETCLIENTSCommand(SOCKET s1)
		{
			vSendCommand(s1, std::string("GETCLIENTS"));
			std::string strMessage = strReceiveMessage(s1);

			std::list<ClientInfo*> plstAddresses;

			char* pcAddresses = new char[strMessage.size()];//as 1 char space for null is also required
			::strcpy(pcAddresses, strMessage.c_str());
			char *pcNextToken = NULL;
			char *pcNextInsideToken = NULL;

			char cSep[1];
			cSep[0] = '\n';

			char cInsideSep[1];
			cInsideSep[0] = ' ';
			do //test this carefully
			{
				strtok_s(pcAddresses, cSep, &pcNextToken);
				//cout << addresses << endl;
				ClientInfo *info = new ClientInfo();
				std::vector<char*> itemData;
				pcAddresses += '\0';
				do
				{
					strtok_s(pcAddresses, cInsideSep, &pcNextInsideToken);
					//cout << addresses << endl;
					itemData.push_back(pcAddresses);
					pcAddresses += '\0';
					pcAddresses = pcNextInsideToken;
				} while (pcNextToken[0] != '\0');

				info->m_strIP = std::string(itemData[0]);
				if (std::stoi(itemData[1]) == 0)
					info->m_boIsApproved = false;
				else
					info->m_boIsApproved = true;
				plstAddresses.push_back(info);

				pcAddresses = pcNextToken;
			} while (pcNextToken[0] != '\0');
			return plstAddresses;
		}

		void TCPClient::vProcessSTARTCommand(SOCKET s1)
		{
			vSendCommand(s1, std::string("START"));
		}

		void TCPClient::vApprove(char ipToApprove[])
		{
			vSendCommand(m_sockClient, std::string("APPROVE"));
			vSendCommand(m_sockClient, std::string(ipToApprove));
		}
		int TCPClient::iStop()
		{
			vSendCommand(m_sockClient, std::string("STOP"));
			if (::closesocket(m_sockClient) != 0){
				printf("iStop: closesocket failed for socket %d!", m_sockClient);
				return -1;
			}
			WSACleanup();//not completely sure if needed
			return 0;
		}

		std::string TCPClient::strReceiveMessage(SOCKET s1)
		{
			int iReceived = 0;
			char cSize[4] = { 0 };
			int iSize = 0;
			iReceived = ::recv(s1, cSize, sizeof(cSize), 0);
			if (iReceived <= 0){
				printf("TCPClient::ReceiveMessage: recv failed for socket %d!", s1);
				return "";
			}
			iSize = *((int*)cSize);
			char *pcMsg = new char[iSize];
			iReceived = ::recv(s1, pcMsg, iSize, 0);
			if (iReceived <= 0){
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

		void TCPClient::vSetConfig(sSimpleConfig stConfiguration)
		{
			vSendCommand(m_sockClient, std::string("SENDCONFIG"));

			std::string configString = "";
			configString += std::to_string(stConfiguration.lWidth);
			configString += std::to_string(stConfiguration.lHeight);
			configString += std::to_string(stConfiguration.lDisplayWidth);
			configString += std::to_string(stConfiguration.lDisplayHeight);
			configString += std::to_string(stConfiguration.iVideoType);
			vSendCommand(m_sockClient, configString);
		}

		int TCPClient::iDisconnect()
		{
			vSendCommand(m_sockClient, std::string("DISCONNECT"));
			return 0;
		}

		void TCPClient::vStart()
		{
			vProcessSTARTCommand(m_sockClient);
		}

		void TCPClient::vStopReceived()
		{
			if (iStop() != 0)
				printf("StopReceived: closesocket failed for socket %d!", m_sockClient);
			emit StopPipeline();
		}

        int TCPClient::iConnectToSession(char IP[], bool bIsOperator)
        {
			WORD wVersionRequested;
			WSADATA wsaData;
			wVersionRequested = MAKEWORD(2, 2);
			WSAStartup(wVersionRequested, &wsaData);//not completely sure if needed

            m_sockClient = oConnectToServer("12345", IP);
            m_sockAlert = oConnectToServer("12345", IP);
            vProcessCONNECTCommand(m_sockClient, bIsOperator);
            TCPClientWrapper* pWrapper = new TCPClientWrapper(m_sockAlert);
            QThread* pThread = new QThread;
            pWrapper->moveToThread(pThread);
			connect(pWrapper, SIGNAL(SetConfigForPipeline(sSimpleConfig)),
				this, SIGNAL(SetConfigForPipeline(sSimpleConfig)), Qt::DirectConnection);
			connect(pWrapper, SIGNAL(Stop()), this, SLOT(StopReceived()), Qt::DirectConnection);
            pThread->start();
            return 0;
        }

        int TCPClient::iRemoveClient(char cClientID[])//now ip used as id
        {
            vProcessREMOVECommand(m_sockClient, cClientID);
            return 0;
        }

        std::list<ClientInfo*> TCPClient::oGetClientList()
        {
            return oProcessGETCLIENTSCommand(m_sockClient);
        }

        TCPClient::TCPClient()
        {
        }

        TCPClient::~TCPClient()
        {
        }

        TCPClientWrapper::TCPClientWrapper(SOCKET s1)
        {
            m_oS1 = s1;
        }

        TCPClientWrapper::~TCPClientWrapper()
        {
        }

        std::string TCPClientWrapper::strReceiveMessage(SOCKET s1)
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

        void TCPClientWrapper::vProcessSTOPCommand(SOCKET s1)
        {
            emit Stop();
        }

        void TCPClientWrapper::vProcessSENDCONFIGCommand(SOCKET s1)
        {
			std::string message = strReceiveMessage(s1);
			sSimpleConfig* stConfig = new sSimpleConfig();

			if (message != "INVALID")
			{
				std::list<char*> parameters = oParseMessageToList(message);
				stConfig->lWidth = std::atol(parameters.front());
				parameters.pop_front();
				stConfig->lHeight = std::atol(parameters.front());
				parameters.pop_front();
				stConfig->lDisplayWidth = std::atol(parameters.front());
				parameters.pop_front();
				stConfig->lDisplayHeight = std::atol(parameters.front());
				parameters.pop_front();
				stConfig->iVideoType = std::atoi(parameters.front());
				parameters.pop_front();
			}

			emit SetConfigForPipeline(*stConfig);
        }

		std::list<char*> TCPClientWrapper::oParseMessageToList(std::string message)
		{
			std::list<char*> plstAddresses;
			//todo: delete the allocated memory
			char* pcAddresses = new char[message.size()];//as 1 char space for null is also required
			::strcpy(pcAddresses, message.c_str());
			char cSep[2];
			char *pcNextToken = NULL;
			cSep[0] = '\n';
			::strtok_s(pcAddresses, cSep, &pcNextToken);
			std::cout << pcAddresses << std::endl;
			plstAddresses.push_back(pcAddresses);
			while (pcNextToken[0] != '\0')
			{
				pcAddresses = pcNextToken;
				if (::strtok_s(pcAddresses, cSep, &pcNextToken) != NULL){
					std::cout << pcAddresses << std::endl;
					plstAddresses.push_back(pcAddresses);
				}
			}
			return plstAddresses;
		}

        void TCPClientWrapper::Process()
        {
            for (;;)
            {
                std::string message = strReceiveMessage(m_oS1);

                if (message == "STOP")
                    vProcessSTOPCommand(m_oS1);
                if (message == "SENDCONFIG")
                    vProcessSENDCONFIGCommand(m_oS1);
            }
        }
    }
}