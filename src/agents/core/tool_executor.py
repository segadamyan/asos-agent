"""
Tool execution utilities for SimpleAgent.
This module handles tool invocation, error handling, and result processing.
"""

from typing import List

from agents.providers.models.base import Message, ToolCallResult
from agents.tools.exceptions import ToolExecutionError
from agents.tools.invoker import ToolInvoker
from agents.utils.logs.config import logger


class ToolExecutor:
    """Handles tool invocation and result processing for SimpleAgent"""

    def __init__(self, invoker: ToolInvoker, agent_name: str):
        self.invoker = invoker
        self.agent_name = agent_name

    async def invoke_tools(self, message: Message) -> List[ToolCallResult]:
        """Invoke all tools in a message and return results"""
        tool_results = []

        for tool_call in message.tool_calls:
            result = await self.invoke_single_tool(tool_call)
            tool_results.append(result)

        return tool_results

    async def invoke_single_tool(self, tool_call) -> ToolCallResult:
        """Invoke a single tool and handle the result"""
        logger.info(f"Invoking tool: {tool_call.tool_name} \n Arguments: {tool_call.tool_args}")
        try:
            result = await self.invoker.invoke(tool_call.tool_name, tool_call.tool_args, tool_call.tool_call_id)

            tool_result = ToolCallResult(
                success=True,
                content=str(result),
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_args=tool_call.tool_args,
            )

        except ToolExecutionError as exc:
            tool_result = ToolCallResult(
                success=False,
                tool_call_id=tool_call.tool_call_id,
                content=f"Error occurred running - {tool_call}\n resulted in: {exc}",
                tool_name=tool_call.tool_name,
                tool_args=tool_call.tool_args,
            )

        logger.info(
            f"Invocation completed for: {tool_call.tool_name} \n "
            f"Arguments: {tool_call.tool_args} \n Result: {tool_result.content}"
        )

        return tool_result
