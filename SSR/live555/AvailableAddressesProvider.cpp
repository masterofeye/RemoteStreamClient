#include "AvailableAddressesProvider.h"


AvailableAddressesProvider::AvailableAddressesProvider(int addressNumber)
{
	m_iAddressNumber = addressNumber;
	m_pAvailableAddresses = new in_addr[m_iAddressNumber];
	m_bAddressListCreated = false;
}

in_addr * AvailableAddressesProvider::GetAllAvailableAddresses(UsageEnvironment * env)
{
	if (!m_bAddressListCreated)
	{
		struct in_addr address;
		for (int i = 0; i < m_iAddressNumber; i++)
		{
			address.s_addr = chooseRandomIPv4SSMAddress(*env);
			m_pAvailableAddresses[i] = address;
		}
		m_bAddressListCreated = true;
	}
	return m_pAvailableAddresses;
}

AvailableAddressesProvider::~AvailableAddressesProvider()
{
	//todo: delete m_pAvailableAddresses
}
