from multiprocessing import Process, Pipe, cpu_count


def child_process_wrapper(runner_func, child_conn, *args):
    output = runner_func(*args)
    child_conn.send([output])
    child_conn.close()


def multiprocessing_handler(runner_func=None, args_list=None):
    outputs_list = []

    if not runner_func or not args_list:
        return outputs_list

    print('Num processes required:', len(args_list))
    print('Num processors available:', cpu_count())

    # Create a list to keep all processes
    processes = []
    # Create a list to keep connections
    connections = []

    # Create a process per argument list
    for arg_list in args_list:
        # Create a pipe for communication
        parent_conn, child_conn = Pipe(duplex=False)
        connections.append((parent_conn, child_conn))
        # Create the process, pass args and connection object
        process = Process(target=child_process_wrapper, args=(runner_func, child_conn, *arg_list))
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Receive outputs from all child processes
    for parent_conn, child_conn in connections:
        child_conn.close() # Ensure child_conn is not open in parent process so parent_conn.recv() doesn't block
        outputs_list.append(parent_conn.recv()[0])

    # Make sure that all processes have finished
    for process in processes:
        process.join()

    return outputs_list
