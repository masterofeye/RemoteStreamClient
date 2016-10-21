#pragma once
#include "IAvailableAddressesProvider.h"

//#define WIN32_LEAN_AND_MEAN
//#include <Windows.h>
//#include <WinSock2.h>
//#include <Ws2tcpip.h>

#include "GroupsockHelper.hh"
#include <liveMedia.hh>
#include <UsageEnvironment.hh>



class AvailableAddressesProvider :
	public IAvailableAddressesProvider
{
private:
	in_addr *m_pAvailableAddresses;
	int m_iAddressNumber;
	bool m_bAddressListCreated;

public:
	virtual in_addr *GetAllAvailableAddresses(UsageEnvironment *env);

	AvailableAddressesProvider(int addressNumber);
	~AvailableAddressesProvider();
};

