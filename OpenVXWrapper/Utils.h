#pragma once

#include <string>
#include <stdint.h>
#include <windows.h>

#ifdef REMOTE_EXPORT
#define REMOTE_API __declspec(dllexport)
#else
#define REMOTE_API __declspec(dllimport)
#endif

//STL Export Issues
#pragma warning (disable : 4083)
#pragma warning (disable : 4251)

namespace RW{
	enum class tenStatus
	{
		nenSuccess,
		nenError,
	};

    namespace CORE{
		enum class tenModule
		{
			enVideoGrabber,
			enGraphic,
			enEncoder,
			enDecoder,
			enRecord,
			enSend,
			enReceive,
            enPlayback,
		};

		enum class tenSubModule
		{
			nenVideoGrabber_WC,
			nenVideoGrabber_FG_USB,
			nenVideoGrabber_FG_PCI,
			nenVideoGrabber_SIMU,
			nenGraphic_Color,
			nenEncode_NVIDIA,
			nenEncode_INTEL,
			nenDecoder_NVIDIA,
			nenDecoder_INTEL,
			nenStream_Simple,
			nenStream_Productive,
			nenReceive_Simple,
			nenReceive_Productive,
            nenPlayback_Simple
		};

		enum class tenStatusVX
		{
			nen1,
			nen2,
		};

        typedef struct rwvx_callback_t
        {
            void* ptr;

        }rwvx_callback;

        enum class tenDir
        {
            nenInput,
            nenOutput,
            nenBidirection
        };

		typedef struct stModuleVersion
		{
			uint8_t u8Major;
            uint8_t u8Minor;
		}tstModuleVersion;

		typedef struct stBoderMode
		{
			int BorderMode; //TODO define and use real enum
			uint32_t u32ConstantValue;

		}tstBoderMode;

		typedef struct stKernelInfo
		{
			int KernelNumber;
			std::string KernelName;
		}tstKernelInfo;



		typedef struct stPerfomance
		{
			uint64_t tmp;
			uint64_t beg;
			uint64_t end;
			uint64_t sum;
			uint64_t avg;
			uint64_t min;
			uint64_t num;
			uint64_t max;
		}tstPerfomance;

        class Util
        {
        public: 
            static std::string getexepath()
            {
                TCHAR NPath[MAX_PATH];
                GetModuleFileName(NULL, NPath, MAX_PATH);
                std::wstring wstring(NPath);
                return std::string (wstring.begin(), wstring.end());
            }
        };
	}
}
