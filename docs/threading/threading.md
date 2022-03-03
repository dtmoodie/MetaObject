Our goal is to have a threading programming API similar to CUDA where we can have asynchronous work streams that we can launch work on.
The the goal is to be able to do something like the following where we have a stream of execution of primary tasks as well as we can push work to
an additional stream for asynchronous execution.

We need to handle three types of tasks
1. A task with no further dependencies, which requires no synchronization or ensuring of order of operations.
2. A task with further dependencies which requires synchronization
3. Events which we just need to operate on the newest event, IE updating the user interface with the newest value

Currently all tasks get submitted to a prioritized work queue, each thread has a set of work queues for each prioritization level
When a cpu side stream is created it is created with the following properties:
- name
- host execution priority
- host allocator
