"""Fixtures compartidos de los tests."""

import pytest


@pytest.fixture(scope="session")
def datos():
    """Datos reales cargados una sola vez por sesión de tests."""
    from agente.datos import cargar_datos

    return cargar_datos()


# Los tests `requiere_clave` llaman al modelo real y gastan créditos: corren solo si se piden
# explícitamente (`make test-clave` o PRUEBAS_CON_CLAVE=1), aunque haya una clave en `.env`.
def pytest_collection_modifyitems(config, items):
    import os

    import pytest

    if os.environ.get("PRUEBAS_CON_CLAVE") == "1":
        return
    saltear = pytest.mark.skip(reason="usa el modelo real: correr con make test-clave (PRUEBAS_CON_CLAVE=1)")
    for item in items:
        if "requiere_clave" in item.keywords:
            item.add_marker(saltear)
