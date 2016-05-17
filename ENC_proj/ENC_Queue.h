#pragma once

#include "common/inc/NvHWEncoder.h"

namespace RW{
	namespace ENC{

		template<class T>
		class ENC_Queue
		{
			friend class ENC_CudaInterop;
		private:
			T** m_pBuffer;
			unsigned int m_uSize;
			unsigned int m_uPendingCount;
			unsigned int m_uAvailableIdx;
			unsigned int m_uPendingndex;
		private:
			ENC_Queue() : m_pBuffer(NULL), m_uSize(0), m_uPendingCount(0), m_uAvailableIdx(0),
				m_uPendingndex(0)
			{
			}

			~ENC_Queue()
			{
				delete[] m_pBuffer;
			}

			bool Initialize(T *pItems, unsigned int uSize)
			{
				m_uSize = uSize;
				m_uPendingCount = 0;
				m_uAvailableIdx = 0;
				m_uPendingndex = 0;
				m_pBuffer = new T *[m_uSize];
				for (unsigned int i = 0; i < m_uSize; i++)
				{
					m_pBuffer[i] = &pItems[i];
				}
				return true;
			}

			T * GetAvailable()
			{
				T *pItem = NULL;
				if (m_uPendingCount == m_uSize)
				{
					return NULL;
				}
				pItem = m_pBuffer[m_uAvailableIdx];
				m_uAvailableIdx = (m_uAvailableIdx + 1) % m_uSize;
				m_uPendingCount += 1;
				return pItem;
			}

			T* GetPending()
			{
				if (m_uPendingCount == 0)
				{
					return NULL;
				}

				T *pItem = m_pBuffer[m_uPendingndex];
				m_uPendingndex = (m_uPendingndex + 1) % m_uSize;
				m_uPendingCount -= 1;
				return pItem;
			}
		};
	}
}

