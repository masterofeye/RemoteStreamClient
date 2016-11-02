#pragma once
#include <QObject>

#include <spdlog\spdlog.h>

extern "C" {
#include <VX\vx.h>
}

namespace RW
{
	namespace CORE
	{
		class Graph : public QObject
		{
		private:
			std::shared_ptr<spdlog::logger> m_Logger;

		public:
			Graph();
			~Graph();
		};
	}
}
