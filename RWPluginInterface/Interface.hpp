#ifndef HEADERFILE_H
#define HEADERFILE_H

#include <QtPlugin>
#include <Utils.h>


namespace RW{
	namespace CORE{

        typedef struct stInitialiseControlStruct
		{
        }tstInitialiseControlStruct;

        typedef struct stControlStruct
        {
        }tstControlStruct;

        typedef struct stDeinitialiseControlStruct
        {
        }tstDeinitialiseControlStruct;

		class AbstractModule : public QObject
		{
			Q_OBJECT
		public:
			virtual ~AbstractModule() {};
			public slots:
			virtual CORE::tstModuleVersion ModulVersion() = 0;
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