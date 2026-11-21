"""Fixtures compartidos de los tests."""

import pytest


@pytest.fixture(scope="session")
def datos():
    """Datos reales cargados una sola vez por sesión de tests."""
    from agente.datos import cargar_datos

    return cargar_datos()
