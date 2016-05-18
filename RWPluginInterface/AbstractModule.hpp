#ifndef HEADERFILE_H
#define HEADERFILE_H

#include <QtPlugin>
#include <Utils.h>
#include "spdlog\spdlog.h"


namespace RW{
	namespace CORE{
		typedef enum tenColorSpace
		{
			nenRGB,
			nenUnknown
		};

        typedef struct stInitialiseControlStruct
		{
			int nFrameWidth;
			int nFrameHeight;
			int nFPS;
			int nNumberOfFrames;
			tenColorSpace enColorSpace;
        }tstInitialiseControlStruct;

		typedef enum tenControlCommands
		{
			tenGrabCommand,
			tenInvalidCommand
		};

		typedef struct stControlStruct
        {
			tenControlCommands enCommand;
			void *pData;
			size_t nDataLength;
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