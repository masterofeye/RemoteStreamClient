#include "Node.h"
#include "Context.h"
#include "Kernel.h"
#include "Graph.h"
#include "AbstractModule.hpp"

#include "HighResolution\HighResClock.h"

#define TRACE_PERFORMANCE 1
namespace RW
{
	namespace CORE
	{
        Node::Node(Graph const *CurrentGraph, Kernel const *Kernel2Connect, std::shared_ptr<spdlog::logger> Logger) :
            m_Graph((*CurrentGraph)()),
            m_Kernel((*Kernel2Connect)()),
            m_Initialize(false),
            m_Logger(Logger)
		{
            CreateNode();
			AssignNodeCallback();
		}

		Node::~Node()
		{
			if (m_Initialize && m_Node)
			{
                for (std::vector<vx_reference>::iterator it = m_ListOfReferences.begin(); it != m_ListOfReferences.end(); it++)
                {
                    int type = 0;
                    vx_status status = vxQueryReference(*it, VX_REF_ATTRIBUTE_TYPE, &type, sizeof(int));
                    if (status != VX_SUCCESS)
                    {
                        //TODO LOG
                    }
                    else
                    {
                        switch (type)
                        {
                        case VX_TYPE_SCALAR:
                            {
                                vx_scalar s = (vx_scalar)(*it);
                                vxReleaseScalar(&s);
                            }
                            break;
                        default:
                            //TODO Log
                            break;
                        }
                    }
                }

				vx_status currentStat = vxReleaseNode(&m_Node);
				if (currentStat != VX_SUCCESS)
				{
                    m_Logger->error("Couldn't release Node.");
				}
                m_Logger->debug("Node released.");
			}
		}

		CORE::tenStatusVX Node::Status()
		{
			if (m_Initialize && m_Node)
			{

				vx_status currentStatus = VX_FAILURE;
				vx_status ret = VX_FAILURE;

				ret = vxQueryNode(m_Node, VX_NODE_ATTRIBUTE_STATUS, &currentStatus, sizeof(vx_status));
				if (ret == VX_SUCCESS)
				{
					//TODO Cast
					//return static_cast<typename std::underlying_type<CORE::tenStatusVX>::type>(currentStatus);
                    return CORE::tenStatusVX::nen1;
				}
				else
				{
					//TODO error
					return CORE::tenStatusVX::nen1;
				}
			}
			else
			{
				return CORE::tenStatusVX::nen1;
			}
		}

		tstPerfomance Node::Performance()
		{
			tstPerfomance perfomance;
			vx_perf_t perfomanceVX;
			if (m_Initialize && m_Node)
			{
				vx_status ret = VX_FAILURE;

				ret = vxQueryNode(m_Node, VX_NODE_ATTRIBUTE_STATUS, &perfomanceVX, sizeof(vx_perf_t));
				if (ret == VX_SUCCESS)
				{
					VX2RWPerformance(perfomanceVX, &perfomance);
					return perfomance;
				}
				else
				{
					//TODO error
					return perfomance;
				}
			}
			else
			{
				return perfomance;
			}
		}

		tstBoderMode Node::BorderMode()
		{
            //TODO real values
            tstBoderMode border = { 1, 2 };
			vx_border_mode_t borderVX;
			if (m_Initialize && m_Node)
			{
				vx_status ret = VX_FAILURE;

				ret = vxQueryNode(m_Node, VX_NODE_ATTRIBUTE_STATUS, &borderVX, sizeof(vx_border_mode_t));
				if (ret == VX_SUCCESS)
				{
					border.BorderMode = borderVX.mode;
					border.u32ConstantValue = borderVX.constant_value;
					return border;
				}
				else
				{
					//TODO error
					return border;
				}
			}
			else
			{
				return border;
			}
		
		}

		size_t Node::LocalDataSize(){
            //TODO implemantation
            return 0;
        }
		void* Node::LocalDataPtr(){
            //TODO implemantation
            return nullptr;
        }

		tstBoderMode Node::SetBorderMode(){
            tstBoderMode borderMode = { 0, 0 }; 
            return borderMode;
        }

        tenStatus Node::SetParameterByIndex(uint32_t Index, void* Value, size_t StructSize, Context const *CurrentContext)
        {
            if (CurrentContext == nullptr)
            {
                m_Logger->alert("No valid context");
                return tenStatus::nenError;
            }
            vx_enum en = vxRegisterUserStruct((*CurrentContext)(), StructSize);
            vx_array testArray = vxCreateArray((*CurrentContext)(), en, 1);
            vx_status res = vxAddArrayItems(testArray, 1, Value, StructSize);
            if (res != VX_SUCCESS)
                return tenStatus::nenError;

            if (vxSetParameterByIndex(m_Node, Index, (vx_reference)testArray) != VX_SUCCESS)
            {
                m_Logger->error("Couldn't add parameter to Node. ") << "Index: " << Index;
                return tenStatus::nenError;
            }
            m_Logger->debug("Parameter added to Node (Index: ") << Index << ")";
            return tenStatus::nenSuccess;
        }

        tenStatus Node::SetParameterByIndex(uint32_t Index, uint8_t Value, Context const *CurrentContext)
        {
            if (CurrentContext == nullptr)
            {
                m_Logger->alert("No valid context");
                return tenStatus::nenError;
            }

            vx_scalar scalar = vxCreateScalar((*CurrentContext)(), VX_TYPE_UINT8, &Value);
            if (scalar == nullptr)
            {
                m_Logger->error("Couldn't create scalar");
                return tenStatus::nenError;
            }
            m_ListOfReferences.push_back((vx_reference)scalar);

            if(vxSetParameterByIndex(m_Node, Index, (vx_reference)scalar) == VX_SUCCESS)
            {
                m_Logger->error("Couldn't add paraemter to node.");
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }

        tenStatus Node::SetParameterByIndex(uint32_t Index, std::string Value, Context const* CurrentContext)
        {

            if (CurrentContext == nullptr)
            {
                m_Logger->alert("No valid context");
                return tenStatus::nenError;
            }

            vx_scalar scalar = vxCreateScalar((*CurrentContext)(), VX_TYPE_UINT8, &Value);
            if (scalar == nullptr)
            {
                m_Logger->error("Couldn't create scalar");
                return tenStatus::nenError;
            }
            m_ListOfReferences.push_back((vx_reference)scalar);

            if (vxSetParameterByIndex(m_Node, Index, (vx_reference)scalar) == VX_SUCCESS)
            {
                m_Logger->error("Couldn't add paraemter to node.");
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }



        tenStatus Node::CreateNode()
        {
            tenStatus status = tenStatus::nenError;
            m_Node = vxCreateGenericNode(m_Graph,m_Kernel);
            vx_status vxStatus = vxGetStatus((vx_reference)m_Node);
            if (vxStatus == VX_SUCCESS)
            {
                m_Logger->debug("Node created.");
                m_Initialize = true;
                return status;
            }
            else
            {
                m_Logger->error("Couldn't create Node.");
                m_Initialize = false;
                return tenStatus::nenSuccess;
            }
        }

        tenStatus Node::AssignNodeCallback()
        {
            vx_status status = vxAssignNodeCallback(m_Node, NodeCallback);
            if (status != VX_SUCCESS)
            {
                m_Logger->error("Assign node callback failed");
                return tenStatus::nenError;
            }
            return tenStatus::nenSuccess;
        }

        vx_action VX_CALLBACK Node::NodeCallback(vx_node Node)
        {
#ifdef TRACE_PERFORMANCE
            auto t1 = RW::CORE::HighResClock::now();
#endif
            vx_parameter param = vxGetParameterByIndex(Node, 2);
            if (param)
            {
                vx_status status = VX_FAILURE;
                vx_array kernenArray, controlStructArray, controlStructArrayNext;
                vx_parameter param[] = { vxGetParameterByIndex(Node, 0), vxGetParameterByIndex(Node, 2) };
                status = vxQueryParameter((vx_parameter)param[0], VX_PARAMETER_ATTRIBUTE_REF, &kernenArray, sizeof(kernenArray));
                if (status != VX_SUCCESS)
                {
                    return VX_FAILURE;
                }
                vx_size size;

                Kernel *kernel = nullptr;
                vxAccessArrayRange(kernenArray, 0, 1, &size, (void**)&kernel, VX_READ_AND_WRITE);

                status = vxQueryParameter((vx_parameter)param[1], VX_PARAMETER_ATTRIBUTE_REF, &controlStructArray, sizeof(controlStructArray));
                if (status != VX_SUCCESS)
                {
                    return VX_FAILURE;
                }

                RW::CORE::tstControlStruct *controlStruct = nullptr;
                vxAccessArrayRange(controlStructArray, 0, 1, &size, (void**)&controlStruct, VX_READ_AND_WRITE);
                RW::CORE::tstControlStruct *mycontrolStruct = kernel->GetControlStruct();

                vx_node nextnode = kernel->Node()->NexttNode();

                vx_parameter param2 = vxGetParameterByIndex(nextnode, 2);
                status = vxQueryParameter((vx_parameter)param2, VX_PARAMETER_ATTRIBUTE_REF, &controlStructArrayNext, sizeof(controlStructArrayNext));
                if (status != VX_SUCCESS)
                {
                    return VX_FAILURE;
                }
                RW::CORE::tstControlStruct *controlStructNext = nullptr;
                vxAccessArrayRange(controlStructArrayNext, 0, 1, &size, (void**)&controlStructNext, VX_READ_AND_WRITE);

                controlStruct->UpdateData(&controlStructNext, RW::CORE::tenSubModule::nenGraphic_Crop);
                //memcpy(mycontrolStruct, controlStruct, size);
                vxCommitArrayRange(controlStructArrayNext, 0, 1, controlStructNext);
            }


        
#ifdef TRACE_PERFORMANCE
			auto t2 = RW::CORE::HighResClock::now();
			auto t3 = RW::CORE::HighResClock::diffMilli(t1, t2).count();
#endif
            return VX_ACTION_CONTINUE;
        }
	}
}


