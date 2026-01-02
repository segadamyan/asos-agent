"""
AsyncThinkAgent - Advanced agent with worker pool and fork/join capabilities.

This agent extends SimpleAgent to provide parallel task execution through worker agents.
It includes fork and join tools

Architecture:
    ┌─────────────────┐
    │ AsyncThinkAgent │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Worker Pool   │
    │  ┌──────────┐   │
    │  │ Worker 1 │   │
    │  │ Worker 2 │   │
    │  │ Worker N │   │
    │  └──────────┘   │
    └─────────────────┘
             │
    ┌────────┴────────┐
    │  Fork/Join      │
    │  Tools          │
    └─────────────────┘
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents.core.simple import SimpleAgent
from agents.providers.models.base import (
    GenerationBehaviorSettings,
    History,
    IntelligenceProviderConfig,
)
from agents.tools.base import ToolDefinition
from agents.utils.generic import load_prompt
from agents.utils.logs.config import logger


class WorkerResult(BaseModel):
    """Represents the result from a worker agent task."""

    task_id: str
    result: Any = None
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class WorkerPool:
    """Manages a pool of worker agents for parallel task execution."""

    def __init__(self, size: int, agent_config: dict):
        self.size = size
        self.agent_config = agent_config
        self.workers: List[SimpleAgent] = []
        self.busy_workers: Dict[str, SimpleAgent] = {}
        self.task_results: Dict[str, WorkerResult] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the worker pool with SimpleAgent instances."""
        if self._initialized:
            return

        for i in range(self.size):
            worker = SimpleAgent(name=f"Worker-{i}", **self.agent_config)
            self.workers.append(worker)

        self._initialized = True
        logger.info(f"Initialized worker pool with {self.size} workers")

    async def get_available_worker(self) -> Optional[SimpleAgent]:
        """Get an available worker from the pool."""
        for worker in self.workers:
            if worker.name not in self.busy_workers:
                return worker
        return None

    async def execute_task(self, task_id: str, task: str, gbs: Optional[GenerationBehaviorSettings] = None) -> str:
        """Execute a task on an available worker. If no workers available, wait for one."""
        max_wait_time = 60  # seconds
        wait_interval = 0.1  # seconds
        waited_time = 0

        worker = await self.get_available_worker()
        while worker is None and waited_time < max_wait_time:
            await asyncio.sleep(wait_interval)
            waited_time += wait_interval
            worker = await self.get_available_worker()

        if worker is None:
            error_msg = (
                f"No workers available after waiting {max_wait_time}s. All {len(self.workers)} workers are busy."
            )
            worker_result = WorkerResult(task_id=task_id, result=None, success=False, error=error_msg)
            self.task_results[task_id] = worker_result
            logger.error(f"Task {task_id} failed: {error_msg}")
            return task_id

        self.busy_workers[task_id] = worker
        logger.info(f"Task {task_id} started on worker {worker.name}")

        try:
            task_worker = await worker.fork(keep_history=False)
            result = await task_worker.answer_to(task, gbs)

            worker_result = WorkerResult(task_id=task_id, result=result.content)
            self.task_results[task_id] = worker_result

            logger.info(f"Task {task_id} completed successfully on worker {worker.name}")
            return task_id

        except Exception as e:
            error_msg = str(e)
            worker_result = WorkerResult(task_id=task_id, result=None, success=False, error=error_msg)
            self.task_results[task_id] = worker_result

            logger.error(f"Task {task_id} failed: {error_msg}")
            return task_id

        finally:
            if task_id in self.busy_workers:
                del self.busy_workers[task_id]
                logger.info(f"Worker {worker.name} released from task {task_id}")

    def get_task_result(self, task_id: str) -> Optional[WorkerResult]:
        """Get the result of a completed task."""
        return self.task_results.get(task_id)

    def cleanup_task_result(self, task_id: str):
        """Clean up task result to free memory."""
        if task_id in self.task_results:
            del self.task_results[task_id]


class AsyncThinkAgent(SimpleAgent):
    """
    Advanced agent that extends SimpleAgent with worker pool capabilities.

    Provides fork/join tools for parallel task execution and result aggregation.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        history: History,
        ip_config: IntelligenceProviderConfig,
        worker_pool_size: int = 10,
        tools: List[ToolDefinition] = None,
        _gbs: Optional[GenerationBehaviorSettings] = None,
        enable_parallel_execution: bool = True,
        max_invocations_count: Optional[int] = None,
        fallback_ip_configs: Optional[List[IntelligenceProviderConfig]] = None,
        max_provider_retries: int = 2,
    ):
        self.worker_config = {
            "system_prompt": system_prompt,
            "history": history,
            "ip_config": ip_config,
            "tools": tools or [],
            "_gbs": _gbs,
            "enable_parallel_execution": enable_parallel_execution,
            "max_invocations_count": max_invocations_count,
            "fallback_ip_configs": fallback_ip_configs,
            "max_provider_retries": max_provider_retries,
        }

        base_prompt = system_prompt
        fork_join_explanation = load_prompt("fork_join_capabilities_prompt").format(worker_pool_size=worker_pool_size)

        self._system_prompt = f"{base_prompt}\n\n{fork_join_explanation}"

        self.worker_pool = WorkerPool(worker_pool_size, self.worker_config)

        fork_join_tools = self._create_fork_join_tools()
        if tools is None:
            tools = []
        tools.extend(fork_join_tools)

        super().__init__(
            name=name,
            system_prompt=self._system_prompt,
            history=history,
            ip_config=ip_config,
            tools=tools,
            _gbs=_gbs,
            enable_parallel_execution=enable_parallel_execution,
            max_invocations_count=max_invocations_count,
            fallback_ip_configs=fallback_ip_configs,
            max_provider_retries=max_provider_retries,
        )

        self.active_tasks: Dict[str, asyncio.Task] = {}

    def _create_fork_join_tools(self) -> List[ToolDefinition]:
        """Create the fork and join tools."""

        async def fork_task(task: str) -> str:
            """Fork a task to a worker agent for background execution."""
            if not self.worker_pool._initialized:
                await self.worker_pool.initialize()

            task_id = str(uuid.uuid4())[:4]
            while task_id in self.active_tasks or task_id in self.worker_pool.task_results:
                task_id = str(uuid.uuid4())[:4]

            available_workers = len(self.worker_pool.workers) - len(self.worker_pool.busy_workers)
            logger.info(
                f"Forking task {task_id}. Available workers: {available_workers}/{len(self.worker_pool.workers)}"
            )

            async_task = asyncio.create_task(self.worker_pool.execute_task(task_id, task, self._gbs))
            self.active_tasks[task_id] = async_task

            logger.info(f"Task {task_id} queued for execution: {task[:50]}...")
            return f"Task forked with ID: {task_id}. Available workers: {available_workers}/{len(self.worker_pool.workers)}"

        async def join_task(task_id: str) -> str:
            """Join result from a single forked task."""
            task_id = task_id.strip()

            if not task_id:
                return "No task ID provided. Please provide a valid task ID."

            if task_id in self.active_tasks:
                try:
                    await self.active_tasks[task_id]
                    del self.active_tasks[task_id]
                except Exception as e:
                    logger.error(f"Task {task_id} execution failed: {e}")

            result = self.worker_pool.get_task_result(task_id)
            if result:
                self.worker_pool.cleanup_task_result(task_id)

                if result.success:
                    return f"Task {task_id} completed successfully:\n{result.result}"
                else:
                    return f"Task {task_id} failed: {result.error}"
            else:
                return f"Task {task_id} not found or never executed."

        fork_tool = ToolDefinition(
            name="fork_task",
            description="Fork a task to a worker agent for background execution. Use this to parallelize work. Each task will have a unique task_id returned for joining.",
            args_description={
                "task": "The task or query to execute in the background",
            },
            tool=fork_task,
            args_schema={},
        )

        join_tool = ToolDefinition(
            name="join_task",
            description="Join and collect result from a previously forked task by its ID.",
            args_description={"task_id": "Task ID to join and collect result from"},
            tool=join_task,
            args_schema={},
        )

        return [fork_tool, join_tool]

    async def answer_to(self, query: str, gbs: Optional[GenerationBehaviorSettings] = None):
        """
        Override answer_to to automatically cleanup resources after completion.
        """
        try:
            result = await super().answer_to(query, gbs)
            return result
        finally:
            await self.shutdown()

    async def answer_to_without_cleanup(self, query: str, gbs: Optional[GenerationBehaviorSettings] = None):
        """
        Answer a query without automatic cleanup. Use this if you want to keep the worker pool active
        for multiple sequential calls. Remember to call shutdown() manually when done.
        """
        return await super().answer_to(query, gbs)

    async def shutdown(self):
        """Shutdown the agent and clean up resources."""
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
            self.active_tasks.clear()

        self.worker_pool.workers.clear()
        self.worker_pool.busy_workers.clear()
        self.worker_pool.task_results.clear()

        logger.info("AsyncThinkAgent shutdown complete")

    async def get_worker_pool_status(self) -> dict:
        """Get current status of the worker pool."""
        return {
            "total_workers": len(self.worker_pool.workers),
            "busy_workers": len(self.worker_pool.busy_workers),
            "available_workers": len(self.worker_pool.workers) - len(self.worker_pool.busy_workers),
            "active_tasks": len(self.active_tasks),
            "active_task_ids": list(self.active_tasks.keys()),
            "pending_results": len(self.worker_pool.task_results),
            "pending_result_ids": list(self.worker_pool.task_results.keys()),
            "busy_worker_details": {task_id: worker.name for task_id, worker in self.worker_pool.busy_workers.items()},
        }


if __name__ == "__main__":

    async def main():
        """Test AsyncThinkAgent initialization without API calls."""
        print("Testing AsyncThinkAgent initialization...")

        ip_config = IntelligenceProviderConfig(provider_name="gemini", version="gemini-2.5-flash")

        gbs = GenerationBehaviorSettings(web_search=False, thinking=False)

        agent = AsyncThinkAgent(
            name="TestAgent",
            system_prompt=f"Test agent with parallel processing capabilities. Current date: {datetime.today().strftime('%Y/%m/%d')}. You can fork multiple tasks simultaneously to test the worker pool.",
            history=History(),
            ip_config=ip_config,
            worker_pool_size=4,
            _gbs=gbs,
        )

        status = await agent.get_worker_pool_status()
        print(f"Worker pool status: {status}")

        response = await agent.answer_to(
            """Please use the fork_task tool to execute multiple tasks in parallel:
            1. Use fork_task with 'What is 2+2?'
            2. Use fork_task with 'What is the capital of France?'
            3. Use fork_task with 'What is 10*5?'
            4. Use fork_task with 'What year is it now?'
            5. Use fork_task with 'What is the square root of 16?'
            6. Use fork_task with 'What is the largest planet?'
            7. Use fork_task with 'What color is the sky?'
            8. Use fork_task with 'What is 100/4?'
            
            This creates 8 tasks but we only have 4 workers - let's see how the system handles queuing!
            Then tell me what tools you have available and how many tasks you forked.
            
            Then join all the tasks and provide me with their results.
            """
        )
        print(f"Response: {response.content[:600]}...")

    asyncio.run(main())
