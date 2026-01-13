"""
Medical and Biology Tools
"""

from typing import List, Optional

from agents.tools.base import ToolDefinition


async def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    Calculate Body Mass Index (BMI).

    Args:
        weight_kg: Weight in kilograms
        height_m: Height in meters

    Returns:
        BMI value and category
    """
    try:
        bmi = weight_kg / (height_m**2)

        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        return f"BMI: {bmi:.2f} ({category})"
    except Exception as e:
        return f"Error calculating BMI: {str(e)}"


async def calculate_dosage(
    patient_weight_kg: float,
    dose_per_kg: float,
    concentration_mg_per_ml: Optional[float] = None,
) -> str:
    """
    Calculate medication dosage.

    Args:
        patient_weight_kg: Patient weight in kilograms
        dose_per_kg: Dose per kilogram in mg/kg
        concentration_mg_per_ml: Concentration of medication in mg/mL (optional)

    Returns:
        Total dose and volume if concentration provided
    """
    try:
        total_dose_mg = patient_weight_kg * dose_per_kg

        result = f"Total dose: {total_dose_mg:.2f} mg"

        if concentration_mg_per_ml is not None:
            volume_ml = total_dose_mg / concentration_mg_per_ml
            result += f"\nVolume: {volume_ml:.2f} mL (at {concentration_mg_per_ml} mg/mL)"

        return result
    except Exception as e:
        return f"Error calculating dosage: {str(e)}"


async def calculate_heart_rate_zones(
    age: int,
    resting_heart_rate: Optional[int] = None,
) -> str:
    """
    Calculate heart rate zones using Karvonen formula.

    Args:
        age: Age in years
        resting_heart_rate: Resting heart rate in bpm (optional)

    Returns:
        Heart rate zones
    """
    try:
        max_hr = 220 - age

        if resting_heart_rate is not None:
            # Karvonen formula
            hr_reserve = max_hr - resting_heart_rate
            zones = {
                "Recovery (50-60%)": int(resting_heart_rate + hr_reserve * 0.5),
                "Aerobic (60-70%)": int(resting_heart_rate + hr_reserve * 0.6),
                "Anaerobic (70-80%)": int(resting_heart_rate + hr_reserve * 0.7),
                "VO2 Max (80-90%)": int(resting_heart_rate + hr_reserve * 0.8),
                "Neuromuscular (90-100%)": int(resting_heart_rate + hr_reserve * 0.9),
            }
        else:
            # Simple percentage of max HR
            zones = {
                "Moderate (50-70%)": f"{int(max_hr * 0.5)}-{int(max_hr * 0.7)}",
                "Vigorous (70-85%)": f"{int(max_hr * 0.7)}-{int(max_hr * 0.85)}",
                "Maximum (85-100%)": f"{int(max_hr * 0.85)}-{max_hr}",
            }

        result = f"Maximum Heart Rate: {max_hr} bpm\n\nHeart Rate Zones:\n"
        for zone, value in zones.items():
            result += f"{zone}: {value} bpm\n"

        return result
    except Exception as e:
        return f"Error calculating heart rate zones: {str(e)}"


async def calculate_creatinine_clearance(
    age: int,
    weight_kg: float,
    serum_creatinine: float,
    gender: str = "male",
) -> str:
    """
    Calculate creatinine clearance using Cockcroft-Gault equation.

    Args:
        age: Age in years
        weight_kg: Weight in kilograms
        serum_creatinine: Serum creatinine in mg/dL
        gender: Gender ("male" or "female")

    Returns:
        Creatinine clearance in mL/min
    """
    try:
        gender_factor = 1.0 if gender.lower() == "male" else 0.85
        clearance = ((140 - age) * weight_kg * gender_factor) / (72 * serum_creatinine)
        return f"Creatinine Clearance: {clearance:.2f} mL/min (Cockcroft-Gault equation)"
    except Exception as e:
        return f"Error calculating creatinine clearance: {str(e)}"


async def calculate_fluid_requirements(
    weight_kg: float,
    maintenance_only: bool = True,
) -> str:
    """
    Calculate daily fluid requirements (Holliday-Segar method).

    Args:
        weight_kg: Weight in kilograms
        maintenance_only: If True, calculate maintenance only; if False, include replacement

    Returns:
        Daily fluid requirements in mL
    """
    try:
        if weight_kg <= 10:
            maintenance = weight_kg * 100
        elif weight_kg <= 20:
            maintenance = 1000 + (weight_kg - 10) * 50
        else:
            maintenance = 1500 + (weight_kg - 20) * 20

        result = f"Maintenance fluids: {maintenance:.0f} mL/day"

        if not maintenance_only:
            # Add replacement (typically 1.5x maintenance for moderate losses)
            replacement = maintenance * 1.5
            result += f"\nWith replacement: {replacement:.0f} mL/day"

        return result
    except Exception as e:
        return f"Error calculating fluid requirements: {str(e)}"


def get_medical_tools() -> List[ToolDefinition]:
    """Get all medical tool definitions."""
    return [
        ToolDefinition(
            name="calculate_bmi",
            description="Calculate Body Mass Index (BMI) and category.",
            args_description={
                "weight_kg": "Weight in kilograms",
                "height_m": "Height in meters",
            },
            args_schema={
                "weight_kg": {"type": "number"},
                "height_m": {"type": "number"},
            },
            tool=calculate_bmi,
        ),
        ToolDefinition(
            name="calculate_dosage",
            description="Calculate medication dosage based on patient weight and dose per kg.",
            args_description={
                "patient_weight_kg": "Patient weight in kilograms",
                "dose_per_kg": "Dose per kilogram in mg/kg",
                "concentration_mg_per_ml": "Concentration of medication in mg/mL (optional)",
            },
            args_schema={
                "patient_weight_kg": {"type": "number"},
                "dose_per_kg": {"type": "number"},
                "concentration_mg_per_ml": {"type": "number"},
            },
            tool=calculate_dosage,
        ),
        ToolDefinition(
            name="calculate_heart_rate_zones",
            description="Calculate heart rate zones using Karvonen formula or percentage of max HR.",
            args_description={
                "age": "Age in years",
                "resting_heart_rate": "Resting heart rate in bpm (optional)",
            },
            args_schema={
                "age": {"type": "integer"},
                "resting_heart_rate": {"type": "integer"},
            },
            tool=calculate_heart_rate_zones,
        ),
        ToolDefinition(
            name="calculate_creatinine_clearance",
            description="Calculate creatinine clearance using Cockcroft-Gault equation.",
            args_description={
                "age": "Age in years",
                "weight_kg": "Weight in kilograms",
                "serum_creatinine": "Serum creatinine in mg/dL",
                "gender": "Gender: 'male' or 'female' (default: 'male')",
            },
            args_schema={
                "age": {"type": "integer"},
                "weight_kg": {"type": "number"},
                "serum_creatinine": {"type": "number"},
                "gender": {"type": "string"},
            },
            tool=calculate_creatinine_clearance,
        ),
        ToolDefinition(
            name="calculate_fluid_requirements",
            description="Calculate daily fluid requirements using Holliday-Segar method.",
            args_description={
                "weight_kg": "Weight in kilograms",
                "maintenance_only": "If True, calculate maintenance only; if False, include replacement (default: True)",
            },
            args_schema={
                "weight_kg": {"type": "number"},
                "maintenance_only": {"type": "boolean"},
            },
            tool=calculate_fluid_requirements,
        ),
    ]
