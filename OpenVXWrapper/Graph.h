#pragma once
extern "C"{
#include "rwvx.h"
}

#include "Utils.h"
#include "Node.h"
#include "spdlog\spdlog.h"

namespace RW{
	namespace CORE{
        
    

        class REMOTE_API Graph
		{
		private:
			vx_graph		m_Graph;
			vx_context		m_Context;
			bool			m_Initialize;
			int				m_NumOfNodes;
			tstPerfomance	m_Performance;
            std::shared_ptr<spdlog::logger> m_Logger;
		public:
            Graph(Context *Context, std::shared_ptr<spdlog::logger> Logger);
			~Graph();

			vx_graph operator()() const
            {
                if (m_Initialize)
                    return m_Graph;
                return nullptr;
            }

			inline  bool IsInitialized() const { return m_Initialize; }
			bool IsGraphVerified();
			tenStatus RemoveNodeFromGraph(Node GraphNode);
			tenStatus WaitGraph();
            tenStatus ProcessGraph();
            tenStatus ScheduleGraph();
            tenStatus VerifyGraph();

			int NumOfParamterers();


		private:
            tenStatus Graph::CreateGraph();
		};
	}
}

