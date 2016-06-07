#ifndef HEADERFILE_H
#define HEADERFILE_H

#include <QtPlugin>
#include <Utils.h>
#include "spdlog\spdlog.h"

namespace RW{
    typedef struct stBitStream
    {
        void *pBuffer;
        uint32_t u32Size;
    }tstBitStream;

    namespace CORE{
		
        typedef struct stInitialiseControlStruct
		{
        }tstInitialiseControlStruct;

        typedef struct stControlStruct
        {
            virtual void UpdateData(struct stControlStruct** Data, tenSubModule SubModuleType) = 0;
        }tstControlStruct;

        typedef struct stDeinitialiseControlStruct
        {
        }tstDeinitialiseControlStruct;

		class AbstractModule : public QObject
		{
			Q_OBJECT
        protected:
            std::shared_ptr<spdlog::logger> m_Logger;
		public:
            AbstractModule(std::shared_ptr<spdlog::logger> Logger) : m_Logger(Logger){}
			virtual ~AbstractModule() {};

			public slots:
			virtual CORE::tstModuleVersion ModulVersion() = 0;
            virtual CORE::tenSubModule SubModulType() = 0;
            virtual tenStatus Initialise(tstInitialiseControlStruct *ControlStruct) = 0;
            virtual tenStatus DoRender(tstControlStruct *ControlStruct) = 0;
            virtual tenStatus Deinitialise(tstDeinitialiseControlStruct *ControlStruct) = 0;
		signals:
			virtual void Finished();
		};

	} 

}
	//QT_BEGIN_NAMESPACE
	//	#define FilterInterface_iid "AbstractModule"
	//	Q_DECLARE_INTERFACE(RW::CORE::AbstractModule, FilterInterface_iid)
	//QT_END_NAMESPACE

#endif