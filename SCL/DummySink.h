#pragma once
#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include <stdint.h>
#include "AbstractModule.hpp"

//#ifdef TRACE_PERFORMANCE
//#include "HighResolution\HighResClock.h"
//#endif
//
class DummySink : public MediaSink {
public:
	typedef void (getDataFunc)(void* clientData, uint8_t* buffer, unsigned size);
	static DummySink* createNew(UsageEnvironment& env, getDataFunc* func, void* clientData);
	

private:
	DummySink(UsageEnvironment& env, getDataFunc* func, void* clientData);
	// called only by "createNew()"
	virtual ~DummySink();

	static void afterGettingFrame(void* clientData, unsigned frameSize,
		unsigned numTruncatedBytes,
	struct timeval presentationTime,
		unsigned durationInMicroseconds);
	void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
	struct timeval presentationTime, unsigned durationInMicroseconds);
	getDataFunc* getDataFunction;
	void Initialise(getDataFunc* func, void* clientData);

private:
	// redefined virtual functions:
	virtual Boolean continuePlaying();
	static void onSourceClosure(void* clientData);
	void onSourceClosure();
	void* m_pClientData;
	FILE* file;

private:
	std::chrono::high_resolution_clock::time_point t1;
	std::shared_ptr<spdlog::logger> Logger;
	uint32_t m_u32SizeComplete;

	bool bFirstEntered;
	u_int8_t* bufferToWrite;
	u_int8_t* fReceiveBuffer;
};

