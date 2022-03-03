In the following example we have a helper stream B that works on subtasks while the main streamA handles main tasks, this ensures sequential operation of tasks and subtasks
An important distinction in the following is that stream A and B could be on the same or different threads, thus achieving true parallelism or pseudo parallelism using boost fibers

streamA --- taskA ---- taskB --- taskC -- taskD
                  \               \
                   \               \
streamB ------------ subtaskA ------ subtaskC

Example code of the above:

void subtaskA(IAsyncStream* stream){}

void taskA(IAsyncStream* stream){
  stream->pushWork(&subtaskA);
}

void taskB(IAsyncStream* stream){
}

void subtaskC(IAsyncStream* stream){}

void taskC(IAsyncStream* stream){
  stream->pushWork(&subtaskC);
}

void taskD(IAsyncStream* streamB){
}

int main(){
  IAsyncStreamPtr_t streamA = IAsyncStream::create();
  IAsyncStreamPtr_t streamB = IAsyncStream::create();
  IAsyncStream* streamB_ptr = streamB.get();
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskA(streamB_ptr); });
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskB(streamB_ptr); });
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskC(streamB_ptr); });
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskD(streamB_ptr); });
  streamA->waitForCompletion();
}

Streams can further synchronize with each other to ensure the work in one stream is done before starting work on another stream: IE

streamA --- taskA ---- taskB --- taskC --------- taskD
                  \               \            /
                   \               \          /
streamB ------------ subtaskA ------ subtaskC 

int main(){
  IAsyncStreamPtr_t streamA = IAsyncStream::create();
  IAsyncStreamPtr_t streamB = IAsyncStream::create();
  IAsyncStream* streamB_ptr = streamB.get();
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskA(streamB_ptr); });
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskB(streamB_ptr); });
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskC(streamB_ptr); });
  // This ensures that anything that is enqueued after this point on streamA is executed after anything that has been enqueued on streamB
  streamA->synchronize(streamB_ptr); 
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskD(streamB_ptr); });
  streamA->waitForCompletion();
}

Explicit threading can be done with the following, this ensures that subtasks operate on a separate worker thread:

int main(){
  IAsyncStreamPtr_t streamA = IAsyncStream::create();
  Thread worker_thread;
  IAsyncStreamPtr_t streamB = worker_thread.getStream();
  IAsyncStream* streamB_ptr = streamB.get();
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskA(streamB_ptr); });
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskB(streamB_ptr); });
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskC(streamB_ptr); });
  // This ensures that anything that is enqueued after this point on streamA is executed after anything that has been enqueued on streamB
  streamA->synchronize(streamB_ptr); 
  streamA->pushWork([streamB_ptr](IAsyncStream*){ taskD(streamB_ptr); });
  streamA->waitForCompletion();
}
