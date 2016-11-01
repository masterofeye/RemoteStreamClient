#pragma once
#include <QObject>

#include <spdlog.h>

extern "C" {
#include <VX\vx.h>
}

namespace RW
{
	namespace CORE
	{
		class Context : public QObject
		{
		private:
			std::shared_ptr<spdlog::logger> m_Logger;

		public:
			Context();
			~Context();
		};
	}
}
