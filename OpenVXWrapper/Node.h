#pragma once
extern "C"{
#include "rwvx.h"
}
#include <vector>
#include "Utils.h"
#include "spdlog\spdlog.h"


namespace RW
{
	namespace CORE
	{
        class Context;
        class Graph;
        class Kernel;


        class REMOTE_API Node
		{
		private:
			bool m_Initialize;
			vx_node m_Node;
			vx_node m_NextNode;
            vx_graph m_Graph;
            vx_kernel m_Kernel;
            std::vector<vx_reference> m_ListOfReferences;
            std::shared_ptr<spdlog::logger> m_Logger;




		public:
            Node(Graph const *CurrentGraph, Kernel const *Kernel2Connect, std::shared_ptr<spdlog::logger> Logger);
			~Node();

            inline bool IsInitialized() const { return m_Initialize; }

			vx_node operator()() const
			{
				if (m_Initialize)
					return m_Node;
				return nullptr;
			}

			CORE::tenStatusVX Status();
			tstPerfomance Performance();
			tstBoderMode BorderMode();
            tstBoderMode SetBorderMode();
			size_t LocalDataSize();
			void* LocalDataPtr();

            tenStatus SetParameterByIndex(uint32_t Index, void* Value, size_t StructSize, Context const *CurrentContext);
            tenStatus SetParameterByIndex(uint32_t Index, uint8_t Value, Context const *CurrentContext);
			tenStatus SetParameterByIndex(uint32_t Index, std::string Value, Context const *CurrentContext);

			/*
			*@brief Set a reference of the following node in the graph execution
			*/
			inline void SetNextNode(Node *FollowingNode){ m_NextNode = (*FollowingNode)(); }

        private: 
            tenStatus CreateNode();
            tenStatus AssignNodeCallback();
            static vx_action VX_CALLBACK Node::NodeCallback(vx_node Node);


			static void VX2RWPerformance(vx_perf_t PerfVX, tstPerfomance* PerfRW)
			{
				if (PerfRW != nullptr)
				{
					PerfRW->avg = PerfVX.avg;
					PerfRW->beg = PerfVX.beg;
					PerfRW->end = PerfVX.end;
					PerfRW->max = PerfVX.max;
					PerfRW->min = PerfVX.min;
					PerfRW->num = PerfVX.num;
					PerfRW->sum = PerfVX.sum;
					PerfRW->tmp = PerfVX.tmp;
				}
			}
		};
	}
}

