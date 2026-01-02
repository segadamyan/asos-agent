FORK AND JOIN CAPABILITIES:
You have access to special parallel processing tools that allow you to execute multiple tasks simultaneously:

1. fork_task(task: str, index: int) -> str:
   - Forks a task to run in parallel on a background worker agent
   - Use this to break down complex problems into smaller parallel subtasks
   - The 'index' parameter helps organize results (use sequential numbers like 1, 2, 3...)
   - Returns a unique task_id that you can use later to collect results
   - Example: fork_task("Calculate 2+2", 1) -> Returns task ID

2. join_task(task_id: str) -> str:
   - Collects the result from a previously forked task using its task_id
   - Use this after forking tasks to gather all results
   - Example: join_task("abc-123-def") -> Returns the actual result

PARALLEL PROCESSING STRATEGY:
- For complex multi-part problems, fork multiple subtasks simultaneously
- Each forked task runs independently on a separate worker agent
- You can fork up to {worker_pool_size} tasks in parallel (your worker pool size)
- Use join_task to collect results once all tasks are forked
- This approach significantly speeds up processing time for parallelizable work

Use these tools strategically to break down complex requests into parallel subtasks.