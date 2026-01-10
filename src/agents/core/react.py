"""
ReActAgent implementation that inherits from Agent.
This agent adds ReAct (Reasoning and Acting) pattern capabilities to the Agent base class.
"""

import logging
from typing import List, Optional

from agents.core.agent import Agent
from agents.providers.factory import ProviderFactory
from agents.providers.models.base import (
    GenerationBehaviorSettings,
    History,
    IntelligenceProviderConfig,
    Message,
    RoleEnum,
)
from agents.providers.models.exceptions import ProviderFailureError
from agents.tools.base import ToolDefinition
from agents.utils.generic import load_prompt

logger = logging.getLogger(__name__)


class ReActAgent(Agent):
    """
    ReAct Agent that inherits from Agent and adds ReAct pattern capabilities.

    Enhances Agent with:
    - ReAct system prompt template with stopping conditions
    - <STOP> keyword detection for intelligent termination
    - Final summary generation when stopping
    - Observation generation after tool usage
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        history: History,
        ip_config: IntelligenceProviderConfig,
        tools: List[ToolDefinition] = None,
        enable_parallel_execution: bool = True,
        fallback_ip_configs: Optional[List[IntelligenceProviderConfig]] = None,
        max_provider_retries: int = 2,
        max_iterations: int = 10,
        _gbs: Optional[GenerationBehaviorSettings] = None,
        max_invocations_count: Optional[int] = None,
        token_manager=None,
    ):
        react_template = load_prompt("react_agent_prompt")
        enhanced_system_prompt = react_template.format(system_prompt=system_prompt)

        super().__init__(
            name=name,
            system_prompt=enhanced_system_prompt,
            history=history,
            ip_config=ip_config,
            tools=tools,
            _gbs=_gbs,
            enable_parallel_execution=enable_parallel_execution,
            max_invocations_count=max_invocations_count,
            fallback_ip_configs=fallback_ip_configs,
            max_provider_retries=max_provider_retries,
            token_manager=token_manager,
        )

        self.max_iterations = max_iterations
        self.current_iteration = 0

    def _check_stop_condition(self, message: Message) -> bool:
        """Check if the message content contains the <STOP> keyword indicating completion"""
        if not message.content:
            return False

        content = message.content.strip()
        return "<STOP>" in content

    async def _generate_observation(self):
        """Generate an observation message based on recent tool results"""
        try:
            cloned_history = History()
            cloned_history.messages = self.history.messages.copy()

            observation_prompt = "Based on the tool results above, provide a brief observation about what was accomplished and what information was gathered. Keep it concise and focused on the key findings."
            cloned_history.add_user_message(observation_prompt)

            observation_config = IntelligenceProviderConfig(
                provider_name=self.providers[self._current_provider_index].provider_name,
                version=self.providers[self._current_provider_index].version,
            )
            observation_provider = self._create_observation_provider(observation_config)

            observation_message = await observation_provider.get_response(history=cloned_history)

            if observation_message.content:
                observation_text = f"Observation: {observation_message.content}"
                self.history.add_assistant_message(observation_text)
            else:
                logger.warning("Observation provider returned empty content, attempting fix")
                fixed_message = await self._fix_empty_message(observation_provider, None)
                if fixed_message.content:
                    observation_text = f"Observation: {fixed_message.content}"
                    self.history.add_assistant_message(observation_text)
        except Exception as exc:
            logger.exception(f"Failed to generate observation: {exc}")
            self.history.add_assistant_message("Observation: Tool execution completed.")

    def _create_observation_provider(self, config: IntelligenceProviderConfig):
        """Create a provider specifically for observations"""

        return ProviderFactory().create(
            system_prompt="You are an analytical assistant that provides concise observations about tool execution results. Focus on summarizing what was accomplished and key information gathered.",
            ip_config=config,
            tools=[],
        )

    async def _generate_final_summary(self, query: str) -> Message:
        """Generate a comprehensive final summary using the intelligence provider"""
        cloned_history = History()
        cloned_history.messages = self.history.messages.copy()

        summary_prompt = f"""
Based on all the actions and reasoning conducted above, please provide a comprehensive final answer to the original query: "{query}"

Please provide:
1. A clear summary of what was accomplished
2. Key findings or results from the actions taken
3. Any relevant information discovered through tool usage
4. Your final answer or recommendation based on all the work performed

Execution Statistics:
- Iterations completed: {self.current_iteration}
- Maximum iterations allowed: {self.max_iterations}

Provide a well-structured, comprehensive response that synthesizes all the work performed into a coherent final answer.
"""

        cloned_history.add_user_message(summary_prompt)

        summary_config = IntelligenceProviderConfig(
            provider_name=self.providers[self._current_provider_index].provider_name,
            version=self.providers[self._current_provider_index].version,
        )
        summary_provider = self._create_summary_provider(summary_config)
        summary_message = await summary_provider.get_response(history=cloned_history)

        if summary_message.content:
            return summary_message
        else:
            logger.warning("Summary provider returned empty content, attempting fix")
            fixed_message = await self._fix_empty_message(summary_provider, None)
            if fixed_message.content:
                return fixed_message
            else:
                raise ProviderFailureError("Failed to fix empty summary response")

    def _create_summary_provider(self, config: IntelligenceProviderConfig):
        """Create a provider specifically for final summaries"""
        return ProviderFactory().create(
            system_prompt="You are a helpful assistant that provides comprehensive, well-structured summaries of completed tasks. Focus on synthesizing information from actions taken and providing clear, actionable final answers.",
            ip_config=config,
            tools=[],
        )

    async def _create_and_return_final_summary(self, query: str) -> Message:
        """Helper method to generate final summary, add to history, and return it"""
        final_summary_message = await self._generate_final_summary(query)
        self.history.add_message(final_summary_message)
        return final_summary_message

    async def _conversation_loop(self, ip, gbs, query: str) -> Optional[Message]:
        """ReAct conversation loop following the original paper pattern"""
        max_attempts = 3
        self.current_iteration = 0

        while self._loop_failed_attempts < max_attempts and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            logger.info(f"Starting ReAct step {self.current_iteration}")

            message = await self._handle_context_overflow(ip.get_response, (), dict(history=self.history, gbs=gbs))

            if message.content and self._check_stop_condition(message):
                self.history.add_message(message)
                return await self._create_and_return_final_summary(query)

            if message.tool_calls:
                self.invocations_count += len(message.tool_calls)

                if self.max_invocations_count is not None and self.invocations_count >= self.max_invocations_count:
                    return await self._handle_invocation_limit_recovery(ip, gbs)

                tool_results = await self._execute_tools(message)
                self.history.add_message(message)

                tool_result_message = Message(
                    content="",
                    role=RoleEnum.TOOL,
                    message_type="tool_result",
                    tool_call_results=tool_results,
                )
                self.history.add_message(tool_result_message)

                # OBSERVE phase
                await self._generate_observation()

                any_failures = any(not result.success for result in tool_results)
                if any_failures:
                    self._loop_failed_attempts += 1
                else:
                    self._loop_failed_attempts = 0

            elif message.content:
                # THINK phase
                self.history.add_message(message)
            else:
                # Empty message
                message = await self._fix_empty_message(ip, gbs)
                if message.content:
                    self.history.add_message(message)

        if self.current_iteration >= self.max_iterations:
            logger.warning("Reached maximum ReAct steps. Generating final summary.")
            return await self._create_and_return_final_summary(query)

        return None
