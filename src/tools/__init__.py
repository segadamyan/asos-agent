"""
Tools Package

This package provides a comprehensive set of tools that agents can use to answer questions
and perform various tasks. Tools are organized by category and can be accessed through
the tools factory.

Tool Categories:
- math_tools: Mathematics, statistics, calculus
- physics_tools: Physics, mechanics, energy, waves, astronomy
- chemistry_tools: Chemistry, molar mass, pH, gas laws, stoichiometry
- computer_science_tools: Algorithms, binary operations, circuits, data structures
- medical_tools: Medical calculations, BMI, dosages, heart rate, fluid requirements
- business_law_tools: Financial ratios, NPV, IRR, econometrics
- logic_philosophy_tools: Logical statements, fallacies, syllogisms
- humanities_tools: Geography, history, astronomy, time zones
"""

from tools.factory import ToolsFactory

__all__ = ["ToolsFactory"]
