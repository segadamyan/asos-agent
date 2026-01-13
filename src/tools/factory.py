"""
Tools Factory

Factory for creating tool sets for different agent types and use cases.
Provides centralized tool management with category-based organization.
"""

from enum import Enum
from typing import Callable, List, Optional, Set

from agents.tools.base import ToolDefinition
from tools.business_law_tools import get_business_law_tools
from tools.chemistry_tools import get_chemistry_tools
from tools.computer_science_tools import get_computer_science_tools
from tools.humanities_tools import get_humanities_tools
from tools.logic_philosophy_tools import get_logic_philosophy_tools
from tools.math_tools import get_math_tools
from tools.medical_tools import get_medical_tools
from tools.physics_tools import get_physics_tools


class ToolCategory(str, Enum):
    """Available tool categories."""

    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    MEDICAL = "medical"
    BUSINESS_LAW = "business_law"
    LOGIC_PHILOSOPHY = "logic_philosophy"
    HUMANITIES = "humanities"


class ToolsFactory:
    """
    Factory for creating and managing tool sets for different agent types.

    Provides methods to:
    - Get tools by category
    - Get all available tools
    - Get tools for specific agent types
    - List available categories
    """

    # Tool category to function mapping
    _TOOL_REGISTRY: dict[str, Callable[[], List[ToolDefinition]]] = {
        ToolCategory.MATH: get_math_tools,
        ToolCategory.PHYSICS: get_physics_tools,
        ToolCategory.CHEMISTRY: get_chemistry_tools,
        ToolCategory.COMPUTER_SCIENCE: get_computer_science_tools,
        ToolCategory.MEDICAL: get_medical_tools,
        ToolCategory.BUSINESS_LAW: get_business_law_tools,
        ToolCategory.LOGIC_PHILOSOPHY: get_logic_philosophy_tools,
        ToolCategory.HUMANITIES: get_humanities_tools,
    }

    @classmethod
    def get_tools_by_category(cls, categories: Optional[List[str]] = None) -> List[ToolDefinition]:
        """
        Get tools for specific categories.

        Args:
            categories: List of category names. If None, returns all tools.

        Returns:
            Deduplicated list of ToolDefinition objects

        Raises:
            ValueError: If an invalid category is provided

        Example:
            >>> tools = ToolsFactory.get_tools_by_category(["math", "physics"])
            >>> all_tools = ToolsFactory.get_tools_by_category()
        """
        if categories is None:
            return cls._get_all_tools()

        # Validate categories
        invalid_categories = [c for c in categories if c not in cls._TOOL_REGISTRY]
        if invalid_categories:
            valid = cls.list_available_categories()
            raise ValueError(f"Invalid categories: {invalid_categories}. Valid categories: {valid}")

        return cls._collect_tools(categories)

    @classmethod
    def get_all_tools(cls) -> List[ToolDefinition]:
        """
        Get all available tools from all categories.

        Returns:
            Deduplicated list of all ToolDefinition objects

        Example:
            >>> all_tools = ToolsFactory.get_all_tools()
            >>> print(f"Total tools: {len(all_tools)}")
        """
        return cls._get_all_tools()

    @classmethod
    def list_available_categories(cls) -> List[str]:
        """
        List all available tool categories.

        Returns:
            Sorted list of category names

        Example:
            >>> categories = ToolsFactory.list_available_categories()
            >>> print(f"Available: {', '.join(categories)}")
        """
        return sorted(cls._TOOL_REGISTRY.keys())

    @classmethod
    def get_category_info(cls) -> dict[str, int]:
        """
        Get information about each category including tool counts.

        Returns:
            Dictionary mapping category names to tool counts

        Example:
            >>> info = ToolsFactory.get_category_info()
            >>> for cat, count in info.items():
            ...     print(f"{cat}: {count} tools")
        """
        return {category: len(get_tools_func()) for category, get_tools_func in cls._TOOL_REGISTRY.items()}

    @classmethod
    def _get_all_tools(cls) -> List[ToolDefinition]:
        """
        Internal method to collect all tools from all categories.

        Returns:
            Deduplicated list of all tools
        """
        return cls._collect_tools(list(cls._TOOL_REGISTRY.keys()))

    @classmethod
    def _collect_tools(cls, categories: List[str]) -> List[ToolDefinition]:
        """
        Internal method to collect and deduplicate tools from specified categories.

        Args:
            categories: List of category names

        Returns:
            Deduplicated list of ToolDefinition objects
        """
        tools: List[ToolDefinition] = []
        seen_names: Set[str] = set()

        for category in categories:
            get_tools_func = cls._TOOL_REGISTRY.get(category)
            if get_tools_func is None:
                continue

            category_tools = get_tools_func()

            for tool in category_tools:
                if tool.name not in seen_names:
                    tools.append(tool)
                    seen_names.add(tool.name)

        return tools
