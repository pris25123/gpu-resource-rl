import random

def generate_job():
    job_type = random.choice(["memory", "compute", "balanced"])

    if job_type == "memory":
        memory = random.uniform(6, 8)
        compute = random.uniform(0.3, 0.5)
        duration = random.randint(4, 6)

    elif job_type == "compute":
        memory = random.uniform(2, 4)
        compute = random.uniform(0.7, 1.0)
        duration = random.randint(3, 5)

    else:  # balanced
        memory = random.uniform(3, 5)
        compute = random.uniform(0.5, 0.7)
        duration = random.randint(3, 5)

    return {
        "type": job_type,
        "memory": memory,
        "compute": compute,
        "duration": duration,
        "remaining_time": duration
    }
