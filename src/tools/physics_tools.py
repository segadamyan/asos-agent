"""
Physics Tools
"""

import math
from typing import List, Optional

from agents.tools.base import ToolDefinition


async def calculate_kinematics(
    initial_velocity: Optional[float] = None,
    final_velocity: Optional[float] = None,
    acceleration: Optional[float] = None,
    time: Optional[float] = None,
    displacement: Optional[float] = None,  # noqa: ARG001
) -> str:
    """
    Calculate kinematics values using equations of motion.

    Args:
        initial_velocity: Initial velocity (v₀) in m/s
        final_velocity: Final velocity (v) in m/s
        acceleration: Acceleration (a) in m/s²
        time: Time (t) in seconds
        displacement: Displacement (s) in meters

    Returns:
        Calculated values or missing parameters
    """
    try:
        results = []

        # v = v₀ + at
        if initial_velocity is not None and acceleration is not None and time is not None:
            final_v = initial_velocity + acceleration * time
            results.append(f"Final velocity: {final_v:.4f} m/s")

        # s = v₀t + ½at²
        if initial_velocity is not None and acceleration is not None and time is not None:
            s = initial_velocity * time + 0.5 * acceleration * time**2
            results.append(f"Displacement: {s:.4f} m")

        # v² = v₀² + 2as
        if initial_velocity is not None and final_velocity is not None and acceleration is not None:
            s = (final_velocity**2 - initial_velocity**2) / (2 * acceleration)
            results.append(f"Displacement: {s:.4f} m")

        # s = ½(v₀ + v)t
        if initial_velocity is not None and final_velocity is not None and time is not None:
            s = 0.5 * (initial_velocity + final_velocity) * time
            results.append(f"Displacement: {s:.4f} m")

        if not results:
            return "Error: Insufficient parameters. Provide at least 3 of: initial_velocity, final_velocity, acceleration, time, displacement"

        return "\n".join(results)

    except Exception as e:
        return f"Error in kinematics calculation: {str(e)}"


async def calculate_force(mass: float, acceleration: float) -> str:
    """
    Calculate force using F = ma.

    Args:
        mass: Mass in kg
        acceleration: Acceleration in m/s²

    Returns:
        Force in Newtons
    """
    try:
        force = mass * acceleration
        return f"Force: {force:.4f} N (F = ma)"
    except Exception as e:
        return f"Error calculating force: {str(e)}"


async def calculate_energy(
    energy_type: str,
    mass: Optional[float] = None,
    velocity: Optional[float] = None,
    height: Optional[float] = None,
    spring_constant: Optional[float] = None,
    displacement: Optional[float] = None,
) -> str:
    """
    Calculate different types of energy.

    Args:
        energy_type: Type of energy: kinetic, potential, spring
        mass: Mass in kg
        velocity: Velocity in m/s
        height: Height in meters
        spring_constant: Spring constant k in N/m
        displacement: Displacement from equilibrium in meters

    Returns:
        Energy value in Joules
    """
    try:
        energy_type = energy_type.lower()

        if energy_type == "kinetic":
            if mass is None or velocity is None:
                return "Error: mass and velocity required for kinetic energy"
            ke = 0.5 * mass * velocity**2
            return f"Kinetic Energy: {ke:.4f} J (KE = ½mv²)"

        elif energy_type == "potential":
            if mass is None or height is None:
                return "Error: mass and height required for potential energy"
            g = 9.81  # gravitational acceleration
            pe = mass * g * height
            return f"Potential Energy: {pe:.4f} J (PE = mgh)"

        elif energy_type == "spring":
            if spring_constant is None or displacement is None:
                return "Error: spring_constant and displacement required for spring energy"
            se = 0.5 * spring_constant * displacement**2
            return f"Spring Energy: {se:.4f} J (E = ½kx²)"

        else:
            return f"Unknown energy type: {energy_type}. Available: kinetic, potential, spring"

    except Exception as e:
        return f"Error calculating energy: {str(e)}"


async def calculate_momentum(mass: float, velocity: float) -> str:
    """
    Calculate momentum using p = mv.

    Args:
        mass: Mass in kg
        velocity: Velocity in m/s

    Returns:
        Momentum in kg·m/s
    """
    try:
        momentum = mass * velocity
        return f"Momentum: {momentum:.4f} kg·m/s (p = mv)"
    except Exception as e:
        return f"Error calculating momentum: {str(e)}"


async def calculate_orbital_velocity(
    mass: float,
    radius: float,
    gravitational_constant: float = 6.67430e-11,
) -> str:
    """
    Calculate orbital velocity using v = √(GM/r).

    Args:
        mass: Mass of central body in kg
        radius: Orbital radius in meters
        gravitational_constant: Gravitational constant G (default: 6.67430e-11)

    Returns:
        Orbital velocity in m/s
    """
    try:
        velocity = math.sqrt(gravitational_constant * mass / radius)
        return f"Orbital Velocity: {velocity:.4f} m/s (v = √(GM/r))"
    except Exception as e:
        return f"Error calculating orbital velocity: {str(e)}"


async def calculate_wave_properties(
    frequency: Optional[float] = None,
    wavelength: Optional[float] = None,
    speed: Optional[float] = None,
) -> str:
    """
    Calculate wave properties using v = fλ.

    Args:
        frequency: Frequency in Hz
        wavelength: Wavelength in meters
        speed: Wave speed in m/s (default: speed of light = 3e8 m/s for electromagnetic waves)

    Returns:
        Calculated wave properties
    """
    try:
        if speed is None:
            speed = 3e8  # speed of light for EM waves

        results = []

        if frequency is not None and wavelength is not None:
            calculated_speed = frequency * wavelength
            results.append(f"Wave speed: {calculated_speed:.4e} m/s")

        if frequency is not None and speed is not None:
            calculated_wavelength = speed / frequency
            results.append(f"Wavelength: {calculated_wavelength:.4e} m")

        if wavelength is not None and speed is not None:
            calculated_frequency = speed / wavelength
            results.append(f"Frequency: {calculated_frequency:.4e} Hz")

        if not results:
            return "Error: Provide at least 2 of: frequency, wavelength, speed"

        return "\n".join(results)

    except Exception as e:
        return f"Error calculating wave properties: {str(e)}"


def get_physics_tools() -> List[ToolDefinition]:
    """Get all physics tool definitions."""
    return [
        ToolDefinition(
            name="calculate_kinematics",
            description="Calculate kinematics values using equations of motion (v = v₀ + at, s = v₀t + ½at², etc.).",
            args_description={
                "initial_velocity": "Initial velocity (v₀) in m/s",
                "final_velocity": "Final velocity (v) in m/s",
                "acceleration": "Acceleration (a) in m/s²",
                "time": "Time (t) in seconds",
                "displacement": "Displacement (s) in meters",
            },
            args_schema={
                "initial_velocity": {"type": "number"},
                "final_velocity": {"type": "number"},
                "acceleration": {"type": "number"},
                "time": {"type": "number"},
                "displacement": {"type": "number"},
            },
            tool=calculate_kinematics,
        ),
        ToolDefinition(
            name="calculate_force",
            description="Calculate force using F = ma.",
            args_description={
                "mass": "Mass in kg",
                "acceleration": "Acceleration in m/s²",
            },
            args_schema={
                "mass": {"type": "number"},
                "acceleration": {"type": "number"},
            },
            tool=calculate_force,
        ),
        ToolDefinition(
            name="calculate_energy",
            description="Calculate different types of energy: kinetic (KE = ½mv²), potential (PE = mgh), spring (E = ½kx²).",
            args_description={
                "energy_type": "Type of energy: kinetic, potential, spring",
                "mass": "Mass in kg",
                "velocity": "Velocity in m/s (for kinetic energy)",
                "height": "Height in meters (for potential energy)",
                "spring_constant": "Spring constant k in N/m (for spring energy)",
                "displacement": "Displacement from equilibrium in meters (for spring energy)",
            },
            args_schema={
                "energy_type": {"type": "string"},
                "mass": {"type": "number"},
                "velocity": {"type": "number"},
                "height": {"type": "number"},
                "spring_constant": {"type": "number"},
                "displacement": {"type": "number"},
            },
            tool=calculate_energy,
        ),
        ToolDefinition(
            name="calculate_momentum",
            description="Calculate momentum using p = mv.",
            args_description={
                "mass": "Mass in kg",
                "velocity": "Velocity in m/s",
            },
            args_schema={
                "mass": {"type": "number"},
                "velocity": {"type": "number"},
            },
            tool=calculate_momentum,
        ),
        ToolDefinition(
            name="calculate_orbital_velocity",
            description="Calculate orbital velocity using v = √(GM/r).",
            args_description={
                "mass": "Mass of central body in kg",
                "radius": "Orbital radius in meters",
                "gravitational_constant": "Gravitational constant G (default: 6.67430e-11)",
            },
            args_schema={
                "mass": {"type": "number"},
                "radius": {"type": "number"},
                "gravitational_constant": {"type": "number"},
            },
            tool=calculate_orbital_velocity,
        ),
        ToolDefinition(
            name="calculate_wave_properties",
            description="Calculate wave properties using v = fλ (wave speed = frequency × wavelength).",
            args_description={
                "frequency": "Frequency in Hz",
                "wavelength": "Wavelength in meters",
                "speed": "Wave speed in m/s (default: 3e8 m/s for EM waves)",
            },
            args_schema={
                "frequency": {"type": "number"},
                "wavelength": {"type": "number"},
                "speed": {"type": "number"},
            },
            tool=calculate_wave_properties,
        ),
    ]
