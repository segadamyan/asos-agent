"""
Humanities Tools
"""

from typing import List, Optional

from agents.tools.base import ToolDefinition


async def calculate_time_difference(
    timezone1: str,
    timezone2: str,
    datetime_str: Optional[str] = None,
) -> str:
    """
    Calculate time difference between two timezones.

    Args:
        timezone1: First timezone (e.g., "UTC", "America/New_York")
        timezone2: Second timezone (e.g., "Europe/London", "Asia/Tokyo")
        datetime_str: Optional datetime string to convert

    Returns:
        Time difference information
    """
    return f"Time difference calculation requires timezone library. Timezone1: {timezone1}, Timezone2: {timezone2}. Use pytz or zoneinfo for accurate calculations."


async def calculate_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    unit: str = "km",
) -> str:
    """
    Calculate distance between two geographic coordinates using Haversine formula.

    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        unit: Unit of distance: "km" or "miles" (default: "km")

    Returns:
        Distance between points
    """
    import math

    try:
        # Haversine formula
        R = 6371.0 if unit == "km" else 3959.0  # Earth radius in km or miles

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        return f"Distance: {distance:.2f} {unit} (Haversine formula)"
    except Exception as e:
        return f"Error calculating distance: {str(e)}"


async def convert_coordinates(
    latitude: float,
    longitude: float,
    format_type: str = "decimal",
) -> str:
    """
    Convert coordinates between decimal and DMS (degrees, minutes, seconds) format.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        format_type: Output format: "decimal" or "dms" (default: "decimal")

    Returns:
        Converted coordinates
    """
    try:
        if format_type.lower() == "dms":

            def to_dms(decimal):
                degrees = int(abs(decimal))
                minutes_float = (abs(decimal) - degrees) * 60
                minutes = int(minutes_float)
                seconds = (minutes_float - minutes) * 60
                direction = (
                    "N" if decimal >= 0 else "S" if abs(decimal) == abs(latitude) else ("E" if decimal >= 0 else "W")
                )
                return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"

            lat_dms = to_dms(latitude)
            lon_dms = to_dms(longitude)
            return f"Latitude: {lat_dms}, Longitude: {lon_dms}"
        else:
            return f"Latitude: {latitude:.6f}°, Longitude: {longitude:.6f}°"
    except Exception as e:
        return f"Error converting coordinates: {str(e)}"


async def calculate_astronomical_distance(
    distance_type: str,
    parallax: Optional[float] = None,
    apparent_magnitude: Optional[float] = None,
    absolute_magnitude: Optional[float] = None,
) -> str:
    """
    Calculate astronomical distances using parallax or magnitude.

    Args:
        distance_type: Type of calculation: "parallax" or "magnitude"
        parallax: Parallax angle in arcseconds (for parallax method)
        apparent_magnitude: Apparent magnitude (for magnitude method)
        absolute_magnitude: Absolute magnitude (for magnitude method)

    Returns:
        Distance in parsecs or light-years
    """
    try:
        if distance_type == "parallax":
            if parallax is None or parallax <= 0:
                return "Error: parallax must be provided and > 0"
            distance_parsecs = 1 / parallax
            distance_ly = distance_parsecs * 3.26156
            return f"Distance: {distance_parsecs:.4f} parsecs ({distance_ly:.4f} light-years)"

        elif distance_type == "magnitude":
            if apparent_magnitude is None or absolute_magnitude is None:
                return "Error: apparent_magnitude and absolute_magnitude required"
            distance_modulus = apparent_magnitude - absolute_magnitude
            distance_parsecs = 10 ** ((distance_modulus + 5) / 5)
            distance_ly = distance_parsecs * 3.26156
            return f"Distance: {distance_parsecs:.4f} parsecs ({distance_ly:.4f} light-years)"

        else:
            return f"Unknown distance_type: {distance_type}. Available: parallax, magnitude"

    except Exception as e:
        return f"Error calculating astronomical distance: {str(e)}"


def get_humanities_tools() -> List[ToolDefinition]:
    """Get all humanities tool definitions."""
    return [
        ToolDefinition(
            name="calculate_time_difference",
            description="Calculate time difference between two timezones.",
            args_description={
                "timezone1": "First timezone (e.g., 'UTC', 'America/New_York')",
                "timezone2": "Second timezone (e.g., 'Europe/London', 'Asia/Tokyo')",
                "datetime_str": "Optional datetime string to convert",
            },
            args_schema={
                "timezone1": {"type": "string"},
                "timezone2": {"type": "string"},
                "datetime_str": {"type": "string"},
            },
            tool=calculate_time_difference,
        ),
        ToolDefinition(
            name="calculate_distance",
            description="Calculate distance between two geographic coordinates using Haversine formula.",
            args_description={
                "lat1": "Latitude of first point",
                "lon1": "Longitude of first point",
                "lat2": "Latitude of second point",
                "lon2": "Longitude of second point",
                "unit": "Unit of distance: 'km' or 'miles' (default: 'km')",
            },
            args_schema={
                "lat1": {"type": "number"},
                "lon1": {"type": "number"},
                "lat2": {"type": "number"},
                "lon2": {"type": "number"},
                "unit": {"type": "string"},
            },
            tool=calculate_distance,
        ),
        ToolDefinition(
            name="convert_coordinates",
            description="Convert coordinates between decimal and DMS (degrees, minutes, seconds) format.",
            args_description={
                "latitude": "Latitude in decimal degrees",
                "longitude": "Longitude in decimal degrees",
                "format_type": "Output format: 'decimal' or 'dms' (default: 'decimal')",
            },
            args_schema={
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "format_type": {"type": "string"},
            },
            tool=convert_coordinates,
        ),
        ToolDefinition(
            name="calculate_astronomical_distance",
            description="Calculate astronomical distances using parallax or magnitude-distance relationship.",
            args_description={
                "distance_type": "Type of calculation: 'parallax' or 'magnitude'",
                "parallax": "Parallax angle in arcseconds (for parallax method)",
                "apparent_magnitude": "Apparent magnitude (for magnitude method)",
                "absolute_magnitude": "Absolute magnitude (for magnitude method)",
            },
            args_schema={
                "distance_type": {"type": "string"},
                "parallax": {"type": "number"},
                "apparent_magnitude": {"type": "number"},
                "absolute_magnitude": {"type": "number"},
            },
            tool=calculate_astronomical_distance,
        ),
    ]
