def display_iteration_progress(iter, niter, task):
    """
    Display the progress of an iteration.

    Args:
        iter (int): The current iteration number.
        niter (int): The total number of iterations.
        task (str): The task being performed.

    Returns:
        None
    """
    per = 100 * iter // niter
    if per != 100 * (iter - 1) // niter:
        print(f'\r{task} {iter}/{niter} ({per}%)', end='')
    if iter == niter:
        print()
