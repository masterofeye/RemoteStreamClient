#pragma once
#include <QObject>

#include <spdlog\spdlog.h>

extern "C" {
#include <VX\vx.h>
}
//Own Headers
#include "../Utils.h"
#include "openvxwrapper_global.h"

namespace RW
{
	namespace CORE
	{
		class OPENVXWRAPPER_EXPORT Node : public QObject
		{
		private:
			std::shared_ptr<spdlog::logger> m_Logger;
			vx_node m_Node;

		public:
			Node();
			~Node();

		private:
			tenStatus CreateNode();
			tenStatus AssignNodeCallback();
			
			static vx_action VX_CALLBACK NodeCallback(vx_node Node);


		};

	}
}