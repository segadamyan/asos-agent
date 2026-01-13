"""
Chemistry Tools
"""

import math
from typing import List, Optional

from agents.tools.base import ToolDefinition


async def calculate_molar_mass(compound: str) -> str:
    """
    Calculate molar mass of a compound.

    Args:
        compound: Chemical formula (e.g., "H2O", "NaCl", "C6H12O6", "Ca(OH)2")

    Returns:
        Molar mass in g/mol
    """
    # Atomic masses (simplified - common elements)
    atomic_masses = {
        "H": 1.008,
        "He": 4.003,
        "Li": 6.941,
        "Be": 9.012,
        "B": 10.81,
        "C": 12.01,
        "N": 14.01,
        "O": 16.00,
        "F": 19.00,
        "Ne": 20.18,
        "Na": 22.99,
        "Mg": 24.31,
        "Al": 26.98,
        "Si": 28.09,
        "P": 30.97,
        "S": 32.07,
        "Cl": 35.45,
        "K": 39.10,
        "Ca": 40.08,
        "Fe": 55.85,
        "Cu": 63.55,
        "Zn": 65.38,
        "Br": 79.90,
        "I": 126.90,
    }

    try:
        import re

        # Remove spaces
        compound = compound.replace(" ", "")

        def parse_formula(formula: str) -> dict:
            """Parse chemical formula and return element counts."""
            element_count = {}

            # Handle parentheses by expanding them
            while "(" in formula:
                # Find innermost parentheses
                match = re.search(r"\(([^()]+)\)(\d*)", formula)
                if not match:
                    break

                inner = match.group(1)
                multiplier = int(match.group(2)) if match.group(2) else 1

                # Expand the parentheses
                expanded = ""
                for element_match in re.finditer(r"([A-Z][a-z]?)(\d*)", inner):
                    element = element_match.group(1)
                    count = int(element_match.group(2)) if element_match.group(2) else 1
                    expanded += element + str(count * multiplier)

                # Replace parentheses with expanded form
                formula = formula[: match.start()] + expanded + formula[match.end() :]

            # Parse elements and their counts
            for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
                element = match.group(1)
                count = int(match.group(2)) if match.group(2) else 1

                if element not in atomic_masses:
                    raise ValueError(f"Unknown element: {element}")

                element_count[element] = element_count.get(element, 0) + count

            return element_count

        # Parse the formula
        element_count = parse_formula(compound)

        # Calculate molar mass
        molar_mass = sum(atomic_masses[element] * count for element, count in element_count.items())

        # Format breakdown
        breakdown = " + ".join(
            f"{count}×{element}({atomic_masses[element]})" for element, count in sorted(element_count.items())
        )

        return f"Molar mass of {compound}: {molar_mass:.3f} g/mol\nBreakdown: {breakdown}"

    except Exception as e:
        return f"Error calculating molar mass: {str(e)}"


async def calculate_ph(h_concentration: Optional[float] = None, oh_concentration: Optional[float] = None) -> str:
    """
    Calculate pH or pOH from concentration.

    Args:
        h_concentration: H⁺ concentration in mol/L
        oh_concentration: OH⁻ concentration in mol/L

    Returns:
        pH or pOH value
    """
    try:
        if h_concentration is not None:
            ph = -math.log10(h_concentration)
            return f"pH: {ph:.4f} (pH = -log[H⁺])"
        elif oh_concentration is not None:
            poh = -math.log10(oh_concentration)
            ph = 14 - poh
            return f"pOH: {poh:.4f}, pH: {ph:.4f} (pH + pOH = 14)"
        else:
            return "Error: Provide either h_concentration or oh_concentration"
    except Exception as e:
        return f"Error calculating pH: {str(e)}"


async def calculate_concentration(moles: float, volume_liters: float) -> str:
    """
    Calculate molarity (concentration) using M = n/V.

    Args:
        moles: Number of moles
        volume_liters: Volume in liters

    Returns:
        Concentration in mol/L
    """
    try:
        concentration = moles / volume_liters if volume_liters != 0 else float("inf")
        return f"Concentration: {concentration:.6f} mol/L (M = n/V)"
    except Exception as e:
        return f"Error calculating concentration: {str(e)}"


async def calculate_ideal_gas_law(
    pressure: Optional[float] = None,
    volume: Optional[float] = None,
    moles: Optional[float] = None,
    temperature: Optional[float] = None,
    r_constant: float = 0.08206,
) -> str:
    """
    Calculate ideal gas law values using PV = nRT.

    Args:
        pressure: Pressure in atm
        volume: Volume in liters
        moles: Number of moles
        temperature: Temperature in Kelvin
        r_constant: Gas constant R (default: 0.08206 L·atm/(mol·K))

    Returns:
        Calculated value
    """
    try:
        results = []

        if pressure is not None and volume is not None and moles is not None:
            temperature = (pressure * volume) / (moles * r_constant)
            results.append(f"Temperature: {temperature:.4f} K")

        if pressure is not None and volume is not None and temperature is not None:
            moles = (pressure * volume) / (r_constant * temperature)
            results.append(f"Moles: {moles:.6f} mol")

        if pressure is not None and moles is not None and temperature is not None:
            volume = (moles * r_constant * temperature) / pressure
            results.append(f"Volume: {volume:.4f} L")

        if volume is not None and moles is not None and temperature is not None:
            pressure = (moles * r_constant * temperature) / volume
            results.append(f"Pressure: {pressure:.4f} atm")

        if not results:
            return "Error: Provide 3 of: pressure, volume, moles, temperature"

        return "\n".join(results)

    except Exception as e:
        return f"Error in ideal gas law calculation: {str(e)}"


async def calculate_stoichiometry(
    balanced_equation: str,
    given_amount: float,
    given_unit: str,
    find: str,
) -> str:
    """
    Calculate stoichiometric relationships (basic implementation).

    Args:
        balanced_equation: Balanced chemical equation (e.g., "2H2 + O2 -> 2H2O")
        given_amount: Amount of given substance
        given_unit: Unit of given amount (moles, grams, liters)
        find: Substance to find in the equation

    Returns:
        Calculated amount
    """
    return f"Stoichiometry calculation requires equation parsing. Equation: {balanced_equation}, Given: {given_amount} {given_unit}, Find: {find}. Use a chemistry library for accurate calculations."


def get_chemistry_tools() -> List[ToolDefinition]:
    """Get all chemistry tool definitions."""
    return [
        ToolDefinition(
            name="calculate_molar_mass",
            description="Calculate molar mass of a chemical compound.",
            args_description={
                "compound": "Chemical formula (e.g., 'H2O', 'NaCl', 'C6H12O6')",
            },
            args_schema={
                "compound": {"type": "string"},
            },
            tool=calculate_molar_mass,
        ),
        ToolDefinition(
            name="calculate_ph",
            description="Calculate pH or pOH from H⁺ or OH⁻ concentration.",
            args_description={
                "h_concentration": "H⁺ concentration in mol/L",
                "oh_concentration": "OH⁻ concentration in mol/L",
            },
            args_schema={
                "h_concentration": {"type": "number"},
                "oh_concentration": {"type": "number"},
            },
            tool=calculate_ph,
        ),
        ToolDefinition(
            name="calculate_concentration",
            description="Calculate molarity (concentration) using M = n/V.",
            args_description={
                "moles": "Number of moles",
                "volume_liters": "Volume in liters",
            },
            args_schema={
                "moles": {"type": "number"},
                "volume_liters": {"type": "number"},
            },
            tool=calculate_concentration,
        ),
        ToolDefinition(
            name="calculate_ideal_gas_law",
            description="Calculate ideal gas law values using PV = nRT.",
            args_description={
                "pressure": "Pressure in atm",
                "volume": "Volume in liters",
                "moles": "Number of moles",
                "temperature": "Temperature in Kelvin",
                "r_constant": "Gas constant R (default: 0.08206 L·atm/(mol·K))",
            },
            args_schema={
                "pressure": {"type": "number"},
                "volume": {"type": "number"},
                "moles": {"type": "number"},
                "temperature": {"type": "number"},
                "r_constant": {"type": "number"},
            },
            tool=calculate_ideal_gas_law,
        ),
        ToolDefinition(
            name="calculate_stoichiometry",
            description="Calculate stoichiometric relationships from balanced chemical equations.",
            args_description={
                "balanced_equation": "Balanced chemical equation (e.g., '2H2 + O2 -> 2H2O')",
                "given_amount": "Amount of given substance",
                "given_unit": "Unit of given amount (moles, grams, liters)",
                "find": "Substance to find in the equation",
            },
            args_schema={
                "balanced_equation": {"type": "string"},
                "given_amount": {"type": "number"},
                "given_unit": {"type": "string"},
                "find": {"type": "string"},
            },
            tool=calculate_stoichiometry,
        ),
    ]
