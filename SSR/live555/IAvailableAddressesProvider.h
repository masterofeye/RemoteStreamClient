#pragma once

//#define WIN32_LEAN_AND_MEAN
//#include <Windows.h>
//#include <WinSock2.h>
//#include <Ws2tcpip.h>

#include <UsageEnvironment.hh>


class IAvailableAddressesProvider
{
public:

	virtual in_addr *GetAllAvailableAddresses(UsageEnvironment *env) = 0;

	virtual ~IAvailableAddressesProvider();
};

