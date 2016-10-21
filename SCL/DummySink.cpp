#include "DummySink.h"
#include <fstream>

#define INDICATOR 4D0E75A3-C76A-44B4-BDE2-CA8EC4C6F00C
#define OUTPUT_FILE "C:\\dummy\\dummyFrames(2).txt"


#define DUMMY_SINK_RECEIVE_BUFFER_SIZE 100000
int counter;
DummySink* DummySink::createNew(UsageEnvironment& env, getDataFunc* func, void* clientData) {
	return new DummySink(env, func, clientData);
}

DummySink::DummySink(UsageEnvironment& env, getDataFunc* func, void* clientData)
	: MediaSink(env)
{
	//todo: probably it would be better to have an "initialize" method //???
	Initialise(func, clientData);
}

DummySink::~DummySink() {
	//fclose(file);
	delete[] fReceiveBuffer;
}

void DummySink::Initialise(getDataFunc* func, void* clientData)
{
	m_pClientData = clientData;
	getDataFunction = func;
	//file = fopen(OUTPUT_FILE, "wb");
	fReceiveBuffer = new u_int8_t[DUMMY_SINK_RECEIVE_BUFFER_SIZE];
	bFirstEntered = true;
}

void DummySink::afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes,
struct timeval presentationTime, unsigned durationInMicroseconds) {
	DummySink* sink = (DummySink*)clientData;
	sink->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime, durationInMicroseconds);
}

// If you don't want to see debugging output for each received frame, then comment out the following line:
//#define DEBUG_PRINT_EACH_RECEIVED_FRAME 1

void DummySink::afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
struct timeval presentationTime, unsigned /*durationInMicroseconds*/) 
{
	//fwrite(fReceiveBuffer, 1, frameSize, file);
	getDataFunction(m_pClientData, fReceiveBuffer, frameSize);
	continuePlaying();
}

void DummySink::onSourceClosure(void* clientData)
{
	DummySink* sink = (DummySink*)clientData;
	sink->onSourceClosure();
}
void DummySink::onSourceClosure()
{
	// Handle the closure for real:
	onSourceClosure();
}


Boolean DummySink::continuePlaying() {
	if (fSource == NULL) return False; // sanity check (should not happen)

	// Request the next frame of data from our input source.  "afterGettingFrame()" will get called later, when it arrives:
	fSource->getNextFrame(fReceiveBuffer, DUMMY_SINK_RECEIVE_BUFFER_SIZE,
		afterGettingFrame, this,
		onSourceClosure, this);
	return True;
}