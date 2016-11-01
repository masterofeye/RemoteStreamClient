#include "Node.h"

extern "C" {
#include <VX\vx.h>
}

namespace RW
{
	namespace CORE
	{

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


		Node::Node()
		{
		}


		Node::~Node()
		{
		}
	}
}