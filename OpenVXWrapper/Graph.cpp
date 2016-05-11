#include "Graph.h"
#include "Context.h"


namespace RW
{
	namespace CORE
	{

        Graph::Graph(Context *CurrentContext, std::shared_ptr<spdlog::logger> Logger) :
            m_Initialize(false),
            m_Logger(Logger)
		{
            if (CurrentContext != nullptr)
			{
                m_Context = (*CurrentContext)();
                CreateGraph();
			}
		}


		Graph::~Graph()
		{
            if (m_Initialize && m_Graph)
            {
                vx_status status = vxReleaseGraph(&m_Graph);
                if (status != VX_SUCCESS)
                {
                    m_Logger->error("Couldn't release graph");
                }
            }
		}

		/*
		* @brief  
		*/
		bool Graph::IsGraphVerified()
		{

			return vxIsGraphVerified(m_Graph);
		}
		tenStatus Graph::RemoveNodeFromGraph(Node GraphNode)
		{
			if(m_Initialize == true && m_Graph != nullptr)
			{
				vx_node node = GraphNode();
				vx_status stat = vxRemoveNode(&node);
				if (stat == VX_SUCCESS)
				{
					return tenStatus::nenSuccess;
				}
				else
				{
                    m_Logger->error("Coulnd't remove node from graph");
					return tenStatus::nenError;
				}

			}
            m_Logger->error("Graph isn't initialized");
			tenStatus res = tenStatus::nenError;
			return res;
		}

		tenStatus Graph::WaitGraph()
		{
			tenStatus res = tenStatus::nenError;
			vx_status status = vxWaitGraph(m_Graph);
            if (status == VX_SUCCESS)
			{
                m_Logger->debug("Wait graph is over.");
                res = tenStatus::nenSuccess;
			}
			else
			{
                m_Logger->error("Wait graph failed.");
			}

			return res;
		}

        tenStatus Graph::ProcessGraph()
        {
            tenStatus res = tenStatus::nenError;
            vx_status status = vxProcessGraph(m_Graph);
            if (status == VX_SUCCESS)
            {
                m_Logger->debug("Graph processed.");
                res = (tenStatus)1;
            }
            else
            {
                res = (tenStatus)0;
                m_Logger->error("Process graph failed.");
            }

            return res;
        }

		tenStatus Graph::VerifyGraph()
		{
			tenStatus res = tenStatus::nenError;
            vx_status status = vxVerifyGraph(m_Graph);
            if (status == VX_SUCCESS)
            {
                m_Logger->debug("Graph is verified.");
                res = tenStatus::nenSuccess;
                return res;
            }
            else
            {
                m_Logger->error("Verify graph failed.");
                m_Initialize = false;
                return res;
            }
			return res;
		}


        tenStatus Graph::ScheduleGraph()
        {
            tenStatus res = tenStatus::nenError;
            vx_status status = vxScheduleGraph(m_Graph);
            if (status == VX_SUCCESS)
            {
                m_Logger->debug("Graph is scheduled.");
                res = tenStatus::nenSuccess;
                return res;
            }
            else
            {
                m_Logger->error("Schedule graph failed.");
                m_Initialize = false;
                return res;
            }
            return res;
        }


		int Graph::NumOfParamterers()
		{
			return 0;
		}

		tenStatus Graph::CreateGraph()
		{
			tenStatus res = tenStatus::nenError;
			vx_status stat = VX_FAILURE;

			m_Graph = vxCreateGraph(m_Context);

			stat = vxGetStatus((vx_reference)m_Context);
			if (stat == VX_SUCCESS)
			{
                m_Logger->debug("Graph created.");
				res = tenStatus::nenSuccess;
				m_Initialize = true;
				return res;
			}
			else
			{
                m_Logger->error("Graph creation failed.");
				m_Initialize = false;
				return res;
			}
		}
	}
}
