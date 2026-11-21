"""CLI: python main.py "pregunta" [--traza]."""

from __future__ import annotations

import json
import sys

from agente.config import FaltaVariableDeEntorno, clave_api, modelo_llm, variable_clave_para


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    mostrar_traza = "--traza" in argv
    argv = [a for a in argv if a != "--traza"]
    if not argv or not argv[0].strip():
        print('Uso: python main.py "tu pregunta" [--traza]', file=sys.stderr)
        return 2
    try:
        clave_api(variable_clave_para(modelo_llm()))
    except FaltaVariableDeEntorno as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    from agente.agente import responder

    respuesta = responder(argv[0])
    # Para recomendaciones, `salida_para_usuario()` es SOLO el JSON.
    print(respuesta.salida_para_usuario())
    if mostrar_traza:
        print(json.dumps(respuesta.traza, ensure_ascii=False, indent=2, default=str), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
