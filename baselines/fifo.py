def schedule_fifo(queue):
    """
    Always return first job.
    """
    if len(queue) == 0:
        return None
    return queue[0]


def schedule_highest_compute(queue):
    """
    Return job with highest compute intensity.
    """
    if len(queue) == 0:
        return None
    return max(queue, key=lambda j: j["compute"])
