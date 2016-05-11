#include "Context.h"

namespace RW
{
	namespace CORE
	{
        Context::Context(std::shared_ptr<spdlog::logger> Logger) :
			m_Initialize(false),
            m_LastStatus(VX_SUCCESS),
            m_Logger(Logger)
		{
            CreateContext();
		}

		Context::~Context()
		{
			if (m_Initialize && m_Context)
			{
				m_LastStatus = vxReleaseContext(&m_Context);
				if (m_LastStatus != VX_SUCCESS)
				{
                    m_Logger->error("Couldn't release Context");
				}
                m_Logger->debug("Context released");
			}
		}

		tenStatus Context::CreateContext()
		{
			tenStatus res = tenStatus::nenError;
			
			m_Context = vxCreateContext();

			m_LastStatus = vxGetStatus((vx_reference)m_Context);
			if (m_LastStatus == VX_SUCCESS)
			{ 
				res = tenStatus::nenSuccess;
                m_Logger->debug("Context created");
				m_Initialize = true;
				return res;
			}
			else
			{
                m_Logger->error("Context couldn't created");
				m_Initialize = false;
			    return res;
			}
			return res;
		}

		uint64_t Context::Vendor()
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return 0;
			}
			else
			{
				uint64_t vendorID = 0;
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_VENDOR_ID, &vendorID, sizeof(uint64_t));
				if (vendorID != 0 && res == VX_SUCCESS)
				{
					return vendorID;
				}
				else
				{
                    m_Logger->error("Couldn't query vendor id");
					return 0;
				}
			}
		}

        uint64_t Context::Version()
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return 0;
			}
			else
			{
				uint64_t version = 0;
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_VERSION, &version, sizeof(uint64_t));
				if (version != 0 && res == VX_SUCCESS)
				{
					return version;
				}
				else
				{
                    m_Logger->error("Couldn't query version id");
					return 0;
				}
			}
		}

		/*
		@brief Queries the context for the number of unique kernels.
		*/
		uint32_t Context::UniqueKernels()
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return 0;
			}
			else
			{
				uint32_t version = 0;
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNELS, &version, sizeof(uint64_t));
				if (version != 0 && res == VX_SUCCESS)
				{
					return version;
				}
				else
				{
                    m_Logger->error("Couldn't query unique kernels amount");
					return 0;
				}
			}
		}

		/*
		@brief Returns the 
		*/
		tstKernelInfo* Context::UniqueKernelTable(uint32_t AmountOfKernel)
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return nullptr;
			}
			else
			{
				 
				tstKernelInfo* aoKernel = new tstKernelInfo[AmountOfKernel];
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNEL_TABLE, aoKernel, AmountOfKernel*sizeof(tstKernelInfo));
				if (aoKernel != nullptr && res == VX_SUCCESS)
				{
					return aoKernel;
				}
				else
				{
                    m_Logger->error("Couldn't query unique kernels");
					return nullptr;
				}
			}
		}

		uint32_t Context::ActiveModules()
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return 0;
			}
			else
			{
				uint32_t modules = 0;
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_MODULES, &modules, sizeof(uint32_t));
				if (modules != 0 && res == VX_SUCCESS)
				{
					return modules;
				}
				else
				{
                    m_Logger->error("Couldn't query modules amount");
					return 0;
				}
			}
		}

		uint32_t Context::ActiveReferences()
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return 0;
			}
			else
			{
				uint32_t references = 0;
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_REFERENCES, &references, sizeof(uint32_t));
				if (references != 0 && res == VX_SUCCESS)
				{
					return references;
				}
				else
				{
                    m_Logger->error("Couldn't query references amount");
					return 0;
				}
			}
		
		}

		std::string Context::ImplementationName()
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return "";
			}
			else
			{
				vx_char *implementatioName = new vx_char[VX_MAX_IMPLEMENTATION_NAME];
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_IMPLEMENTATION, implementatioName, VX_MAX_IMPLEMENTATION_NAME * sizeof(vx_char));
				if (implementatioName != nullptr && res == VX_SUCCESS)
				{
					std::string s(implementatioName);
					delete implementatioName;
					return s;
				}
				else
				{
                    m_Logger->error("Couldn't query implementatio name");
					return "";
				}
			}
		}

		size_t Context::ExtentionSize()
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return 0;
			}
			else
			{
				vx_size extentionSize = 0;
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_EXTENSIONS_SIZE, &extentionSize, sizeof(vx_size));
				if (extentionSize != 0 && res == VX_SUCCESS)
				{
					return extentionSize;
				}
				else
				{
                    m_Logger->error("Couldn't query extention size");
					return 0;
				}
			}
		}

		const std::string Context::Extentions(size_t ExtentionSize)
		{
			if (m_Initialize && m_Context)
			{
                m_Logger->alert("Context not created or initialized");
				return "";
			}
			else
			{
				vx_char *extentionChar = new vx_char[ExtentionSize];
				vx_status res = vxQueryContext(m_Context, VX_CONTEXT_ATTRIBUTE_EXTENSIONS, extentionChar, ExtentionSize * sizeof(vx_char));
				if (extentionChar != nullptr && res == VX_SUCCESS)
				{
					std::string extentionString(extentionChar);
					delete extentionChar;
					return extentionString;
				}
				else
				{
                    m_Logger->error("Couldn't query extention name");
					return "";
				}
			}
		}


		//const size_t Context::MaxConvolutionDimention();
		//const size_t Context::MaxOpticalFlowDimention();
		//const tstBoderMode Context::BorderMode();
		//void Context::SetBorderMode(tstBoderMode BorderMode);

	}
}
