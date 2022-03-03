#include <MetaObject/thread/FiberScheduler.hpp>

#include <MetaObject/core/IAsyncStream.hpp>

#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/fiber/all.hpp>

#include "gtest/gtest.h"

TEST(async_stream, creation)
{
    ASSERT_EQ(mo::IAsyncStream::current(), nullptr);

    auto stream = mo::IAsyncStream::create();
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(mo::IAsyncStream::current(), stream);
}

TEST(async_stream, work)
{
    auto stream = mo::IAsyncStream::create();
    ASSERT_EQ(stream->size(), 0);

    bool work_complete = false;
    stream->pushWork([&work_complete](mo::IAsyncStream*) { work_complete = true; });
    ASSERT_EQ(stream->size(), 1);
    boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
    ASSERT_EQ(work_complete, true);
}

TEST(async_stream, work_sync)
{
    auto stream = mo::IAsyncStream::create();
    ASSERT_EQ(stream->size(), 0);

    bool work_complete = false;
    stream->pushWork([&work_complete](mo::IAsyncStream*)
    {
        work_complete = true;
    });
    ASSERT_EQ(stream->size(), 1);
    stream->synchronize();
    ASSERT_EQ(work_complete, true);
}

TEST(async_stream, current_stream)
{
    auto stream = mo::IAsyncStream::create();
    ASSERT_EQ(stream->size(), 0);

    bool check_passes = false;
    stream->pushWork([&check_passes, stream](mo::IAsyncStream* ref)
    {
        check_passes = (stream == mo::IAsyncStream::current()) && (stream.get() == ref);

    });
    ASSERT_EQ(stream->size(), 1);
    stream->synchronize();
    ASSERT_EQ(check_passes, true);
}



namespace parallel_indepdent_subtasks
{
    int32_t main_task_counter = 0;
    int32_t sub_task_counter = 0;
    bool task_a_called = false;
    bool task_b_called = false;
    bool task_c_called = false;
    bool task_d_called = false;
    bool subtask_a_called = false;
    bool subtask_c_called = false;
    void subtaskA(mo::IAsyncStream* stream){
        sub_task_counter += 10;
        subtask_a_called = true;
    }

    void taskA(mo::IAsyncStream* stream){
        main_task_counter += 1;
        stream->pushWork(&subtaskA);
        task_a_called = true;
    }

    void taskB(mo::IAsyncStream* stream){
        main_task_counter += 2;
        ASSERT_TRUE(task_a_called);
        task_b_called = true;
    }

    void subtaskC(mo::IAsyncStream* stream)
    {
        sub_task_counter *= 3;
        subtask_c_called = true;
        ASSERT_TRUE(subtask_a_called);
    }

    void taskC(mo::IAsyncStream* stream){
        main_task_counter *= 4;
        stream->pushWork(&subtaskC);
        ASSERT_TRUE(task_b_called);
        task_c_called = true;
    }

    void taskD(mo::IAsyncStream* streamB){
        main_task_counter += 5;
        ASSERT_TRUE(task_c_called);
        task_d_called = true;
    }
}

TEST(async_stream, parallel_independent_subtasks)
{
    parallel_indepdent_subtasks::main_task_counter = 0;
    parallel_indepdent_subtasks::sub_task_counter = 0;
    mo::IAsyncStreamPtr_t streamA = mo::IAsyncStream::create();
    mo::IAsyncStream::setCurrent(streamA);

    mo::IAsyncStreamPtr_t streamB = mo::IAsyncStream::create();
    mo::IAsyncStream* streamB_ptr = streamB.get();
    std::shared_ptr<mo::WorkQueue> work_queue_a = streamA->getWorkQueue();
    std::shared_ptr<mo::WorkQueue> work_queue_b = streamB->getWorkQueue();
    ASSERT_TRUE(work_queue_a);
    ASSERT_TRUE(work_queue_b);

    ASSERT_NE(work_queue_a, work_queue_b);

    ASSERT_TRUE(work_queue_a->getScheduler());
    ASSERT_TRUE(work_queue_b->getScheduler());

    streamA->pushWork([streamB_ptr](mo::IAsyncStream*){ parallel_indepdent_subtasks::taskA(streamB_ptr); });
    streamA->pushWork([streamB_ptr](mo::IAsyncStream*){ parallel_indepdent_subtasks::taskB(streamB_ptr); });
    streamA->pushWork([streamB_ptr](mo::IAsyncStream*){ parallel_indepdent_subtasks::taskC(streamB_ptr); });
    streamA->pushWork([streamB_ptr](mo::IAsyncStream*){ parallel_indepdent_subtasks::taskD(streamB_ptr); });
    streamA->waitForCompletion();
    ASSERT_EQ(parallel_indepdent_subtasks::main_task_counter, ((1+2) * 4 + 5));
    ASSERT_EQ(parallel_indepdent_subtasks::sub_task_counter, (10 * 3));
}

namespace parallel_dependent_subtasks
{
    int32_t main_task_counter = 0;
    int32_t sub_task_counter = 0;
    bool task_a_called = false;
    bool task_b_called = false;
    bool task_c_called = false;
    bool task_d_called = false;
    bool subtask_a_called = false;
    bool subtask_c_called = false;

    void subtaskA(){
        sub_task_counter += 10;
        subtask_a_called = true;
    }

    void subtaskC()
    {
        sub_task_counter *= 3;
        subtask_c_called = true;
        ASSERT_TRUE(subtask_a_called);
        ASSERT_TRUE(task_c_called);
    }

    void taskA(){
        main_task_counter += 1;
        task_a_called = true;
    }

    void taskB(){
        main_task_counter += 2;
        ASSERT_TRUE(task_a_called);
        task_b_called = true;
    }

    void taskC(){
        main_task_counter *= 4;
        ASSERT_TRUE(task_b_called);
        task_c_called = true;
    }

    void taskD(){
        main_task_counter += 5;
        ASSERT_TRUE(task_c_called);
        ASSERT_TRUE(subtask_c_called);
        task_d_called = true;
    }
}

TEST(async_stream, parallel_dependent_subtasks)
{
    parallel_dependent_subtasks::main_task_counter = 0;
    parallel_dependent_subtasks::sub_task_counter = 0;
    mo::IAsyncStreamPtr_t stream_a = mo::IAsyncStream::create("stream_a");
    mo::IAsyncStream::setCurrent(stream_a);

    mo::IAsyncStreamPtr_t stream_b = mo::IAsyncStream::create("stream_b");
    mo::IAsyncStream* stream_b_ptr = stream_b.get();
    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskA(); });
    // Encodes that work done on stream b needs to wait for taskA to complete on stream a
    stream_b->synchronize(*stream_a);
    stream_b->pushWork([stream_b_ptr](mo::IAsyncStream*){parallel_dependent_subtasks::subtaskA();});
    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskB(); });
    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskC(); });
    // Encodes that work done on stream b needs to wait for taskC to complete on stream a
    stream_b->synchronize(*stream_a);
    stream_b->pushWork([stream_b_ptr](mo::IAsyncStream*){parallel_dependent_subtasks::subtaskC();});
    // Encodes that work done on stream a needs to wait for subtaskC to complete on stream b
    // The problem is that in the synchronize function, stream_b calls notify before stream a starts to wait
    stream_a->synchronize(*stream_b);// This synchronize breaks the unit test causing it to not finish for some reason
    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskD(); });
    stream_a->waitForCompletion();
    ASSERT_EQ(parallel_dependent_subtasks::main_task_counter, ((1+2) * 4 + 5));
    ASSERT_EQ(parallel_dependent_subtasks::sub_task_counter, (10 * 3));
}

TEST(async_stream, threaded_parallel_dependent_subtasks)
{
    parallel_dependent_subtasks::main_task_counter = 0;
    parallel_dependent_subtasks::sub_task_counter = 0;
    parallel_dependent_subtasks::task_a_called = false;
    parallel_dependent_subtasks::task_b_called = false;
    parallel_dependent_subtasks::task_c_called = false;
    parallel_dependent_subtasks::task_d_called = false;
    parallel_dependent_subtasks::subtask_a_called = false;
    parallel_dependent_subtasks::subtask_c_called = false;

    mo::IAsyncStreamPtr_t stream_a = mo::IAsyncStream::create("stream_a");
    mo::IAsyncStream::setCurrent(stream_a);

    mo::Thread worker_thread;
    mo::IAsyncStreamPtr_t stream_b = worker_thread.asyncStream(std::chrono::seconds(1000));
    mo::IAsyncStream* stream_b_ptr = stream_b.get();

    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskA(); });
    // Encodes that work done on stream b needs to wait for taskA to complete on stream a
    stream_b->synchronize(*stream_a);
    stream_b->pushWork([stream_b_ptr](mo::IAsyncStream*){parallel_dependent_subtasks::subtaskA();});
    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskB(); });
    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskC(); });
    // Encodes that work done on stream b needs to wait for taskC to complete on stream a
    stream_b->synchronize(*stream_a);
    stream_b->pushWork([stream_b_ptr](mo::IAsyncStream*){parallel_dependent_subtasks::subtaskC();});
    // Encodes that work done on stream a needs to wait for subtaskC to complete on stream b
    // The problem is that in the synchronize function, stream_b calls notify before stream a starts to wait
    stream_a->synchronize(*stream_b);// This synchronize breaks the unit test causing it to not finish for some reason
    stream_a->pushWork([stream_b_ptr](mo::IAsyncStream*){ parallel_dependent_subtasks::taskD(); });
    stream_a->waitForCompletion();
    ASSERT_EQ(parallel_dependent_subtasks::main_task_counter, ((1+2) * 4 + 5));
    ASSERT_EQ(parallel_dependent_subtasks::sub_task_counter, (10 * 3));
}
