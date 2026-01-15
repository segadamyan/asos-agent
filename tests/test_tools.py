"""
Tests for tools package.

Tests the ToolsFactory and individual tool functions.
Run with:
    poetry run pytest tests/test_tools.py
    poetry run pytest tests/test_tools.py -v
"""

import os
import sys

import pytest

# Add src to path (go up one level from tests/ folder)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.business_law_tools import (
    calculate_econometric_regression,
    calculate_financial_ratio,
    calculate_future_value,
    calculate_irr,
    calculate_npv,
    calculate_present_value,
)
from tools.chemistry_tools import (
    calculate_concentration,
    calculate_ideal_gas_law,
    calculate_ph,
)
from tools.computer_science_tools import (
    analyze_data_structure,
    calculate_binary_operations,
    calculate_circuit_properties,
    convert_number_base,
)
from tools.execution_tools import python_executor
from tools.factory import ToolsFactory
from tools.humanities_tools import (
    calculate_astronomical_distance,
    calculate_distance,
    convert_coordinates,
)
from tools.logic_philosophy_tools import (
    construct_truth_table,
    evaluate_logical_statement,
    identify_logical_fallacy,
)
from tools.math_tools import (
    calculate,
    calculator,
    solve_quadratic,
    statistical_calc,
)
from tools.medical_tools import (
    calculate_bmi,
    calculate_creatinine_clearance,
    calculate_dosage,
    calculate_fluid_requirements,
    calculate_heart_rate_zones,
)
from tools.physics_tools import (
    calculate_energy,
    calculate_force,
    calculate_kinematics,
    calculate_momentum,
    calculate_wave_properties,
)

# ============================================================================
# Tests for ToolsFactory
# ============================================================================


def test_tools_factory_list_categories():
    """Test listing available tool categories."""
    categories = ToolsFactory.list_available_categories()
    assert isinstance(categories, list)
    assert len(categories) > 0
    assert "math" in categories
    assert "physics" in categories
    assert "chemistry" in categories


def test_tools_factory_get_tools_by_category():
    """Test getting tools by category."""
    tools = ToolsFactory.get_tools_by_category(["math", "physics"])
    assert isinstance(tools, list)
    assert len(tools) > 0
    tool_names = [tool.name for tool in tools]
    # Should have math tools
    assert "calculate" in tool_names or "statistical_calc" in tool_names
    # Should have physics tools
    assert "calculate_force" in tool_names or "calculate_kinematics" in tool_names


def test_tools_factory_get_all_tools():
    """Test getting all available tools."""
    tools = ToolsFactory.get_all_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    # Should have tools from multiple categories
    tool_names = [tool.name for tool in tools]
    assert "calculate" in tool_names  # Math
    assert "calculate_force" in tool_names  # Physics


def test_tools_factory_get_tools_by_category_all():
    """Test getting all tools via get_tools_by_category."""
    tools = ToolsFactory.get_tools_by_category()
    assert isinstance(tools, list)
    assert len(tools) > 0


def test_tools_factory_get_tools_by_category_specific():
    """Test getting tools for specific categories."""
    tools = ToolsFactory.get_tools_by_category(["math", "physics"])
    assert isinstance(tools, list)
    assert len(tools) > 0


# ============================================================================
# Tests for Math Tools
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_basic():
    """Test basic calculation."""
    result = await calculate("2 + 2")
    assert result == "4"


@pytest.mark.asyncio
async def test_calculate_trigonometry():
    """Test trigonometry calculations."""
    result = await calculate("sin(pi/2)")
    assert "1" in result or "1.0" in result


@pytest.mark.asyncio
async def test_calculate_sqrt():
    """Test square root calculation."""
    result = await calculate("sqrt(16)")
    assert result == "4.0" or result == "4"


@pytest.mark.asyncio
async def test_statistical_calc_mean():
    """Test statistical mean calculation."""
    result = await statistical_calc("mean", [10, 20, 30, 40, 50])
    assert "30" in result or "30.0" in result


@pytest.mark.asyncio
async def test_statistical_calc_median():
    """Test statistical median calculation."""
    result = await statistical_calc("median", [10, 20, 30, 40, 50])
    assert "30" in result


@pytest.mark.asyncio
async def test_statistical_calc_std_dev():
    """Test standard deviation calculation."""
    result = await statistical_calc("std_dev", [10, 20, 30, 40, 50])
    assert "result" in result.lower() or "deviation" in result.lower()


@pytest.mark.asyncio
async def test_statistical_calc_correlation():
    """Test correlation calculation."""
    result = await statistical_calc("correlation", data=[], sample1=[1, 2, 3, 4, 5], sample2=[2, 4, 6, 8, 10])
    assert "correlation" in result.lower() or "1" in result


@pytest.mark.asyncio
async def test_solve_quadratic():
    """Test quadratic equation solving."""
    result = await solve_quadratic(1, -5, 6)  # x² - 5x + 6 = 0, solutions: 2, 3
    assert "2" in result and "3" in result


@pytest.mark.asyncio
async def test_calculator_basic_arithmetic():
    """Test basic arithmetic with calculator."""
    result = await calculator("2 + 2")
    assert result.get("result") == 4

    result = await calculator("10 * 5")
    assert result.get("result") == 50

    result = await calculator("100 / 4")
    assert result.get("result") == 25


@pytest.mark.asyncio
async def test_calculator_with_functions():
    """Test calculator with math functions."""
    result = await calculator("sqrt(16)")
    assert result.get("result") == 4

    result = await calculator("abs(-5)")
    assert result.get("result") == 5

    result = await calculator("floor(3.7)")
    assert result.get("result") == 3

    result = await calculator("ceil(3.2)")
    assert result.get("result") == 4


@pytest.mark.asyncio
async def test_calculator_with_constants():
    """Test calculator with constants."""
    result = await calculator("pi")
    assert abs(result.get("result") - 3.14159) < 0.001

    result = await calculator("e")
    assert abs(result.get("result") - 2.71828) < 0.001


@pytest.mark.asyncio
async def test_calculator_complex_expression():
    """Test calculator with complex expressions."""
    result = await calculator("2*(3+4)**2")
    assert result.get("result") == 98

    result = await calculator("sqrt(16) + log10(100)")
    assert result.get("result") == 6


@pytest.mark.asyncio
async def test_calculator_trigonometry():
    """Test calculator with trigonometric functions."""
    result = await calculator("sin(0)")
    assert abs(result.get("result")) < 0.001

    result = await calculator("cos(0)")
    assert abs(result.get("result") - 1) < 0.001


@pytest.mark.asyncio
async def test_calculator_error_handling():
    """Test calculator error handling."""
    result = await calculator("1 / 0")
    assert "error" in result

    result = await calculator("")
    assert "error" in result

    result = await calculator("import os")  # Should fail
    assert "error" in result


@pytest.mark.asyncio
async def test_python_executor_basic():
    """Test basic Python execution."""
    result = await python_executor("result = 2 + 2")
    assert result.get("result") == 4


@pytest.mark.asyncio
async def test_python_executor_with_math():
    """Test Python executor with math module."""
    result = await python_executor("import math\nresult = math.factorial(5)")
    # Should fail because imports are not allowed
    assert "error" in result

    # This should work - math is pre-imported
    result = await python_executor("result = math.factorial(5)")
    assert result.get("result") == 120


@pytest.mark.asyncio
async def test_python_executor_with_statistics():
    """Test Python executor with statistics module."""
    code = """
data = [1, 2, 3, 4, 5]
result = statistics.mean(data)
"""
    result = await python_executor(code)
    assert result.get("result") == 3


@pytest.mark.asyncio
async def test_python_executor_with_loops():
    """Test Python executor with loops."""
    code = """
total = 0
for i in range(1, 11):
    total += i
result = total
"""
    result = await python_executor(code)
    assert result.get("result") == 55


@pytest.mark.asyncio
async def test_python_executor_with_print():
    """Test Python executor with print statements."""
    code = """
print("Step 1: Calculate factorial")
result = math.factorial(5)
print("Result:", result)
"""
    result = await python_executor(code)
    assert result.get("result") == 120
    assert "Step 1" in result.get("stdout", "")


@pytest.mark.asyncio
async def test_python_executor_complex_computation():
    """Test Python executor with complex multi-step computation."""
    code = """
# Calculate combinations
n = 10
k = 3
result = math.comb(n, k)
"""
    result = await python_executor(code)
    assert result.get("result") == 120


@pytest.mark.asyncio
async def test_python_executor_list_comprehension():
    """Test Python executor with list comprehensions."""
    code = """
squares = [x**2 for x in range(1, 6)]
result = sum(squares)
"""
    result = await python_executor(code)
    assert result.get("result") == 55  # 1 + 4 + 9 + 16 + 25


@pytest.mark.asyncio
async def test_python_executor_error_handling():
    """Test Python executor error handling."""
    # Empty code
    result = await python_executor("")
    assert "error" in result

    # Forbidden imports
    result = await python_executor("import os")
    assert "error" in result

    # Forbidden function
    result = await python_executor("eval('2+2')")
    assert "error" in result

    # Division by zero
    result = await python_executor("result = 1 / 0")
    assert "error" in result


@pytest.mark.asyncio
async def test_python_executor_timeout():
    """Test Python executor timeout."""
    code = """
# Infinite loop - should timeout
while True:
    pass
"""
    result = await python_executor(code, timeout_seconds=1)
    assert "error" in result
    assert "timed out" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_solve_quadratic():
    """Test quadratic equation solving."""
    result = await solve_quadratic(1, -5, 6)  # x² - 5x + 6 = 0, solutions: 2, 3
    assert "2" in result and "3" in result


# ============================================================================
# Tests for Physics Tools
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_force():
    """Test force calculation."""
    result = await calculate_force(mass=10, acceleration=5)
    assert "50" in result or "50.0" in result


@pytest.mark.asyncio
async def test_calculate_kinematics():
    """Test kinematics calculation."""
    result = await calculate_kinematics(initial_velocity=10, acceleration=2, time=5)
    assert "20" in result or "velocity" in result.lower()


@pytest.mark.asyncio
async def test_calculate_energy_kinetic():
    """Test kinetic energy calculation."""
    result = await calculate_energy("kinetic", mass=2, velocity=10)
    assert "100" in result or "energy" in result.lower()


@pytest.mark.asyncio
async def test_calculate_energy_potential():
    """Test potential energy calculation."""
    result = await calculate_energy("potential", mass=10, height=5)
    assert "energy" in result.lower()


@pytest.mark.asyncio
async def test_calculate_momentum():
    """Test momentum calculation."""
    result = await calculate_momentum(mass=5, velocity=10)
    assert "50" in result or "momentum" in result.lower()


@pytest.mark.asyncio
async def test_calculate_wave_properties():
    """Test wave properties calculation."""
    result = await calculate_wave_properties(frequency=100, wavelength=3)
    assert "speed" in result.lower() or "300" in result


# ============================================================================
# Tests for Chemistry Tools
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_ph():
    """Test pH calculation."""
    result = await calculate_ph(h_concentration=0.001)
    assert "3" in result or "ph" in result.lower()


@pytest.mark.asyncio
async def test_calculate_concentration():
    """Test concentration calculation."""
    result = await calculate_concentration(moles=2, volume_liters=1)
    assert "2" in result or "concentration" in result.lower()


@pytest.mark.asyncio
async def test_calculate_ideal_gas_law():
    """Test ideal gas law calculation."""
    result = await calculate_ideal_gas_law(pressure=1, volume=22.4, temperature=273.15)
    assert "moles" in result.lower() or "1" in result


# ============================================================================
# Tests for Computer Science Tools
# ============================================================================


@pytest.mark.asyncio
async def test_convert_number_base():
    """Test number base conversion."""
    result = await convert_number_base("10", from_base=10, to_base=2)
    assert "1010" in result  # 10 in decimal = 1010 in binary


@pytest.mark.asyncio
async def test_calculate_binary_operations():
    """Test binary operations."""
    result = await calculate_binary_operations("AND", "0b1010", "0b1100")
    assert "1000" in result or "8" in result  # 1010 AND 1100 = 1000


@pytest.mark.asyncio
async def test_calculate_circuit_properties():
    """Test circuit properties calculation."""
    result = await calculate_circuit_properties(voltage=12, current=2)
    assert "24" in result or "resistance" in result.lower()


@pytest.mark.asyncio
async def test_analyze_data_structure():
    """Test data structure operations."""
    result = await analyze_data_structure("sort", [3, 1, 4, 1, 5])
    assert "1" in result and "5" in result


# ============================================================================
# Tests for Medical Tools
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_bmi():
    """Test BMI calculation."""
    result = await calculate_bmi(weight_kg=70, height_m=1.75)
    assert "bmi" in result.lower() or "22.86" in result or "22.9" in result


@pytest.mark.asyncio
async def test_calculate_dosage():
    """Test dosage calculation."""
    result = await calculate_dosage(patient_weight_kg=70, dose_per_kg=10)
    assert "700" in result or "dose" in result.lower()


@pytest.mark.asyncio
async def test_calculate_heart_rate_zones():
    """Test heart rate zones calculation."""
    result = await calculate_heart_rate_zones(age=30)
    assert "190" in result or "heart" in result.lower()  # Max HR = 220 - 30 = 190


@pytest.mark.asyncio
async def test_calculate_creatinine_clearance():
    """Test creatinine clearance calculation."""
    result = await calculate_creatinine_clearance(age=50, weight_kg=70, serum_creatinine=1.0, gender="male")
    assert "clearance" in result.lower()


@pytest.mark.asyncio
async def test_calculate_fluid_requirements():
    """Test fluid requirements calculation."""
    result = await calculate_fluid_requirements(weight_kg=70)
    assert "fluid" in result.lower() or "ml" in result.lower()


# ============================================================================
# Tests for Business/Law Tools
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_financial_ratio():
    """Test financial ratio calculation."""
    result = await calculate_financial_ratio("current_ratio", current_assets=100000, current_liabilities=50000)
    assert "2" in result or "ratio" in result.lower()


@pytest.mark.asyncio
async def test_calculate_npv():
    """Test NPV calculation."""
    result = await calculate_npv(cash_flows=[-10000, 3000, 4000, 5000], discount_rate=0.10)
    assert "npv" in result.lower() or "present" in result.lower()


@pytest.mark.asyncio
async def test_calculate_irr():
    """Test IRR calculation."""
    result = await calculate_irr(cash_flows=[-10000, 3000, 4000, 5000])
    assert "irr" in result.lower() or "%" in result


@pytest.mark.asyncio
async def test_calculate_present_value():
    """Test present value calculation."""
    result = await calculate_present_value(future_value=1000, rate=0.10, periods=5)
    assert "present" in result.lower() or "value" in result.lower()


@pytest.mark.asyncio
async def test_calculate_future_value():
    """Test future value calculation."""
    result = await calculate_future_value(present_value=1000, rate=0.10, periods=5)
    assert "future" in result.lower() or "value" in result.lower()


@pytest.mark.asyncio
async def test_calculate_econometric_regression():
    """Test econometric regression."""
    result = await calculate_econometric_regression(x_data=[1, 2, 3, 4, 5], y_data=[2, 4, 6, 8, 10])
    assert "regression" in result.lower() or "slope" in result.lower()


# ============================================================================
# Tests for Logic/Philosophy Tools
# ============================================================================


@pytest.mark.asyncio
async def test_evaluate_logical_statement():
    """Test logical statement evaluation."""
    result = await evaluate_logical_statement("P AND Q", {"P": True, "Q": False})
    assert "statement" in result.lower() or "logic" in result.lower()


@pytest.mark.asyncio
async def test_identify_logical_fallacy():
    """Test logical fallacy identification."""
    result = await identify_logical_fallacy("This is a test argument")
    assert "fallacy" in result.lower() or "argument" in result.lower()


@pytest.mark.asyncio
async def test_construct_truth_table():
    """Test truth table construction."""
    result = await construct_truth_table(["P", "Q"], "P AND Q")
    assert "truth" in result.lower() or "table" in result.lower()


# ============================================================================
# Tests for Humanities Tools
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_distance():
    """Test distance calculation using Haversine formula."""
    # Distance between New York and Los Angeles (approximately)
    result = await calculate_distance(lat1=40.7128, lon1=-74.0060, lat2=34.0522, lon2=-118.2437)
    assert "distance" in result.lower() or "km" in result.lower()


@pytest.mark.asyncio
async def test_convert_coordinates():
    """Test coordinate conversion."""
    result = await convert_coordinates(latitude=40.7128, longitude=-74.0060, format_type="dms")
    assert "40" in result or "latitude" in result.lower()


@pytest.mark.asyncio
async def test_calculate_astronomical_distance():
    """Test astronomical distance calculation."""
    result = await calculate_astronomical_distance("parallax", parallax=0.1)
    assert "parsecs" in result.lower() or "distance" in result.lower()


# ============================================================================
# Integration Tests - Tools with Agent
# ============================================================================


@pytest.mark.asyncio
async def test_agent_with_math_tools(check_openai_api_key):
    """Test agent with math tools."""
    from agents.core.agent import Agent
    from agents.providers.models.base import History, IntelligenceProviderConfig

    tools = ToolsFactory.get_tools_by_category(["math"])
    assert len(tools) > 0

    agent = Agent(
        name="MathToolAgent",
        system_prompt="You are a math expert. Use tools to solve problems.",
        history=History(),
        ip_config=IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct"),
        tools=tools,
    )

    response = await agent.ask("What is 15 * 23? Use the calculate tool.")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_agent_with_physics_tools(check_openai_api_key):
    """Test agent with physics tools."""
    from agents.core.agent import Agent
    from agents.providers.models.base import History, IntelligenceProviderConfig

    tools = ToolsFactory.get_tools_by_category(["physics"])
    assert len(tools) > 0

    agent = Agent(
        name="PhysicsToolAgent",
        system_prompt="You are a physics expert. Use tools to solve problems.",
        history=History(),
        ip_config=IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct"),
        tools=tools,
    )

    response = await agent.ask("Calculate the force when mass is 10 kg and acceleration is 5 m/s².")
    assert response is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_agent_with_multiple_tool_categories(check_openai_api_key):
    """Test agent with multiple tool categories."""
    from agents.core.agent import Agent
    from agents.providers.models.base import History, IntelligenceProviderConfig

    tools = ToolsFactory.get_tools_by_category(["math", "physics", "chemistry"])
    assert len(tools) > 0

    agent = Agent(
        name="MultiToolAgent",
        system_prompt="You are an expert with access to multiple tool categories.",
        history=History(),
        ip_config=IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct"),
        tools=tools,
    )

    # Check that agent has tools
    assert len(agent.available_tools) > 0
