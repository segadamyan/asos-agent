"""
Computer Science Tools

"""

from typing import Any, List, Optional

from agents.tools.base import ToolDefinition


async def convert_number_base(number: str, from_base: int, to_base: int) -> str:
    """
    Convert a number from one base to another.

    Args:
        number: Number as string in source base
        from_base: Source base (2-36)
        to_base: Target base (2-36)

    Returns:
        Number in target base
    """
    try:
        decimal = int(number, from_base)

        # Convert to target base
        if to_base == 10:
            return str(decimal)

        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        n = decimal

        if n == 0:
            return "0"

        while n > 0:
            result = digits[n % to_base] + result
            n //= to_base

        return f"{number} (base {from_base}) = {result} (base {to_base})"
    except Exception as e:
        return f"Error converting number: {str(e)}"


async def calculate_binary_operations(
    operation: str,
    operand1: str,
    operand2: Optional[str] = None,
) -> str:
    """
    Perform binary operations (AND, OR, XOR, NOT, shift).

    Args:
        operation: Operation type: AND, OR, XOR, NOT, LEFT_SHIFT, RIGHT_SHIFT
        operand1: First operand (binary string or integer)
        operand2: Second operand (for binary operations, not needed for NOT)

    Returns:
        Result of the operation
    """
    try:
        operation = operation.upper()

        # Convert to integer if binary string
        if isinstance(operand1, str) and operand1.startswith("0b"):
            num1 = int(operand1, 2)
        else:
            num1 = int(operand1)

        if operation == "NOT":
            result = ~num1
            return f"NOT {operand1} = {result} (binary: {bin(result & 0xFFFFFFFF)})"

        if operand2 is None:
            return "Error: operand2 required for binary operations (except NOT)"

        if isinstance(operand2, str) and operand2.startswith("0b"):
            num2 = int(operand2, 2)
        else:
            num2 = int(operand2)

        if operation == "AND":
            result = num1 & num2
        elif operation == "OR":
            result = num1 | num2
        elif operation == "XOR":
            result = num1 ^ num2
        elif operation == "LEFT_SHIFT":
            result = num1 << num2
        elif operation == "RIGHT_SHIFT":
            result = num1 >> num2
        else:
            return f"Unknown operation: {operation}. Available: AND, OR, XOR, NOT, LEFT_SHIFT, RIGHT_SHIFT"

        return f"{operand1} {operation} {operand2} = {result} (binary: {bin(result & 0xFFFFFFFF)})"

    except Exception as e:
        return f"Error in binary operation: {str(e)}"


async def analyze_data_structure(operation: str, data: List[Any], **kwargs: Any) -> str:
    """
    Analyze data structure operations (basic implementation).

    Args:
        operation: Operation type: sort, search, insert, delete, find_min, find_max
        data: List of data elements
        **kwargs: Additional parameters (e.g., value for search/insert/delete)

    Returns:
        Result of the operation
    """
    try:
        operation = operation.lower()

        if operation == "sort":
            sorted_data = sorted(data)
            return f"Sorted: {sorted_data}"

        elif operation == "find_min":
            return f"Minimum: {min(data)}"

        elif operation == "find_max":
            return f"Maximum: {max(data)}"

        elif operation == "search":
            value = kwargs.get("value")
            if value is None:
                return "Error: value parameter required for search"
            if value in data:
                return f"Found {value} at index {data.index(value)}"
            return f"{value} not found"

        elif operation == "insert":
            value = kwargs.get("value")
            position = kwargs.get("position")
            if value is None:
                return "Error: value parameter required for insert"
            if position is not None:
                data.insert(position, value)
            else:
                data.append(value)
            return f"Inserted {value}. New data: {data}"

        elif operation == "delete":
            value = kwargs.get("value")
            if value is None:
                return "Error: value parameter required for delete"
            if value in data:
                data.remove(value)
                return f"Deleted {value}. New data: {data}"
            return f"{value} not found"

        else:
            return f"Unknown operation: {operation}"

    except Exception as e:
        return f"Error in data structure operation: {str(e)}"


async def calculate_circuit_properties(
    voltage: Optional[float] = None,
    current: Optional[float] = None,
    resistance: Optional[float] = None,
    power: Optional[float] = None,
) -> str:
    """
    Calculate electrical circuit properties using Ohm's law and power equations.

    Args:
        voltage: Voltage in Volts (V)
        current: Current in Amperes (A)
        resistance: Resistance in Ohms (Ω)
        power: Power in Watts (W)

    Returns:
        Calculated values
    """
    try:
        results = []

        # V = IR
        if current is not None and resistance is not None:
            v = current * resistance
            results.append(f"Voltage: {v:.4f} V (V = IR)")

        if voltage is not None and current is not None:
            r = voltage / current if current != 0 else float("inf")
            results.append(f"Resistance: {r:.4f} Ω (R = V/I)")

        if voltage is not None and resistance is not None:
            i = voltage / resistance if resistance != 0 else float("inf")
            results.append(f"Current: {i:.4f} A (I = V/R)")

        # P = VI = I²R = V²/R
        if voltage is not None and current is not None:
            p = voltage * current
            results.append(f"Power: {p:.4f} W (P = VI)")

        if current is not None and resistance is not None:
            p = current**2 * resistance
            results.append(f"Power: {p:.4f} W (P = I²R)")

        if voltage is not None and resistance is not None:
            p = voltage**2 / resistance if resistance != 0 else float("inf")
            results.append(f"Power: {p:.4f} W (P = V²/R)")

        if not results:
            return "Error: Provide at least 2 of: voltage, current, resistance, power"

        return "\n".join(results)

    except Exception as e:
        return f"Error calculating circuit properties: {str(e)}"


def get_computer_science_tools() -> List[ToolDefinition]:
    """Get all computer science tool definitions."""
    return [
        ToolDefinition(
            name="convert_number_base",
            description="Convert a number from one base to another (e.g., binary to decimal, hex to binary).",
            args_description={
                "number": "Number as string in source base",
                "from_base": "Source base (2-36)",
                "to_base": "Target base (2-36)",
            },
            args_schema={
                "number": {"type": "string"},
                "from_base": {"type": "integer", "minimum": 2, "maximum": 36},
                "to_base": {"type": "integer", "minimum": 2, "maximum": 36},
            },
            tool=convert_number_base,
        ),
        ToolDefinition(
            name="calculate_binary_operations",
            description="Perform binary operations: AND, OR, XOR, NOT, LEFT_SHIFT, RIGHT_SHIFT.",
            args_description={
                "operation": "Operation type: AND, OR, XOR, NOT, LEFT_SHIFT, RIGHT_SHIFT",
                "operand1": "First operand (binary string or integer)",
                "operand2": "Second operand (not needed for NOT)",
            },
            args_schema={
                "operation": {"type": "string"},
                "operand1": {"type": "string"},
                "operand2": {"type": "string"},
            },
            tool=calculate_binary_operations,
        ),
        ToolDefinition(
            name="analyze_data_structure",
            description="Perform data structure operations: sort, search, insert, delete, find_min, find_max.",
            args_description={
                "operation": "Operation type: sort, search, insert, delete, find_min, find_max",
                "data": "List of data elements",
                "value": "Value for search, insert, or delete operations",
                "position": "Position for insert operation",
            },
            args_schema={
                "operation": {"type": "string"},
                "data": {"type": "array"},
                "value": {"type": "string"},
                "position": {"type": "integer"},
            },
            tool=analyze_data_structure,
        ),
        ToolDefinition(
            name="calculate_circuit_properties",
            description="Calculate electrical circuit properties using Ohm's law (V = IR) and power equations (P = VI = I²R = V²/R).",
            args_description={
                "voltage": "Voltage in Volts (V)",
                "current": "Current in Amperes (A)",
                "resistance": "Resistance in Ohms (Ω)",
                "power": "Power in Watts (W)",
            },
            args_schema={
                "voltage": {"type": "number"},
                "current": {"type": "number"},
                "resistance": {"type": "number"},
                "power": {"type": "number"},
            },
            tool=calculate_circuit_properties,
        ),
    ]
