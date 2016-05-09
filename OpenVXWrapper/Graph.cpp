#include "Graph.h"
#include "Context.h"

namespace RW
{
	namespace CORE
	{

        Graph::Graph(Context *CurrentContext):
            m_Initialize(false)
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
                    //TODO ERROR LOG
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
					return tenStatus::nenError;
				}

			}

			tenStatus res = tenStatus::nenError;
			return res;
		}

		tenStatus Graph::WaitGraph()
		{
			tenStatus res = tenStatus::nenError;
			vx_status status = vxWaitGraph(m_Graph);
			if (status)
			{
				res = (tenStatus)1;
			}
			else
			{
				res = (tenStatus)0;
			}

			return res;
		}

        tenStatus Graph::ProcessGraph()
        {
            tenStatus res = tenStatus::nenError;
            vx_status status = vxProcessGraph(m_Graph);
            if (status)
            {
                res = (tenStatus)1;
            }
            else
            {
                res = (tenStatus)0;
            }

            return res;
        }

		tenStatus Graph::VerifyGraph()
		{
			tenStatus res = tenStatus::nenError;
            vx_status status = vxVerifyGraph(m_Graph);
            if (status == VX_SUCCESS)
            {
                res = tenStatus::nenSuccess;
                return res;
            }
            else
            {
                //Error log
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
                res = tenStatus::nenSuccess;
                return res;
            }
            else
            {
                //Error log
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
				res = tenStatus::nenSuccess;
				m_Initialize = true;
				return res;
			}
			else
			{
				//Error log
				m_Initialize = false;
				return res;
			}
		}
	}
}
