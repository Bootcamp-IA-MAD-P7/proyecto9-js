# Guía: Spec, SDD y Harness aplicados a este proyecto

Este documento explica tres conceptos que ya están en uso en este repositorio
(`.specify/`, `specs/`) y añade la pieza que faltaba (el *harness*), con un
ejemplo real implementado en esta misma rama para que puedas ver el ciclo
completo funcionando de principio a fin.

---

## 1. ¿Qué es una Spec?

Una **spec** (`spec.md`) es un documento que responde a **QUÉ** hay que
construir y **POR QUÉ**, sin entrar todavía en cómo se va a programar.

En este proyecto ya existe una:
[`specs/001-hate-speech-detection/spec.md`](../specs/001-hate-speech-detection/spec.md)

Su estructura:

- **What / Why**: el problema (YouTube no da abasto moderando odio) y la
  solución esperada (clasificador + interfaz consultable).
- **Scope / Out of scope**: qué entra en esta spec concreta y qué se deja
  para más adelante (ensemble, deep learning, deploy... eso son specs
  futuras, ej. `002-...`, `003-...`).
- **Criterios de éxito en formato EARS**: frases con una estructura fija
  (`WHEN ... THE SYSTEM SHALL ...`, `IF ... THEN THE SYSTEM SHALL ...`) que
  son *comprobables*. No dicen "que funcione bien", dicen exactamente qué
  condición de entrada produce qué comportamiento observable.

Ejemplo real del archivo:

> WHEN a raw comments dataset is provided, THE SYSTEM SHALL produce a
> cleaned, preprocessed dataset ready for vectorization.

Esa frase es la que luego se convierte directamente en un test (lo verás en
la sección 4).

## 2. ¿Qué es SDD (Spec-Driven Development)?

Es una metodología de trabajo donde **la especificación manda sobre el
código**, no al revés. El flujo es siempre en este orden:

```
constitution.md  →  spec.md  →  plan.md  →  tasks.md  →  código + tests
   (principios)      (qué/por qué) (cómo, arquitectura) (pasos atómicos)  (implementación)
```

En este repo cada pieza ya existe:

| Fase | Archivo | Responde a |
|---|---|---|
| Constitution | [`.specify/memory/constitution.md`](../.specify/memory/constitution.md) | ¿Qué principios no negociables sigue el proyecto? (ej. controlar overfitting, priorizar lo práctico sobre lo preciso) |
| Specify | [`specs/001-hate-speech-detection/spec.md`](../specs/001-hate-speech-detection/spec.md) | ¿Qué hay que construir y por qué? |
| Plan | [`specs/001-hate-speech-detection/plan.md`](../specs/001-hate-speech-detection/plan.md) | ¿Con qué arquitectura y decisiones técnicas? |
| Tasks | [`specs/001-hate-speech-detection/tasks.md`](../specs/001-hate-speech-detection/tasks.md) | ¿Cuáles son los pasos atómicos, con criterio de aceptación cada uno? |
| Implement | `src/`, `tests/` | El código que hace cumplir literalmente lo anterior |

La clave de SDD es que **nunca se programa "a ojo"**: antes de escribir
`src/data/loader.py`, ya existía la frase EARS en `spec.md` que decía qué
tenía que hacer esa función. El código es una consecuencia de la spec, no al
revés.

Ventaja práctica para un proyecto de bootcamp/equipo: cualquiera puede leer
`tasks.md` y saber exactamente qué falta, y el criterio de "está terminado"
no es subjetivo — es la frase EARS cumplida (o no).

## 3. ¿Qué es el Harness?

Esta es la pieza que añadimos en esta rama. El **harness** es la maquinaria
automática que **verifica continuamente que el código sigue cumpliendo la
spec**, sin depender de que alguien lo compruebe a mano.

Sin harness, SDD se rompe en la práctica: puedes escribir specs preciosas y
luego el código diverge silenciosamente de ellas. El harness cierra ese
ciclo:

```
spec.md (criterio EARS)
      │
      ▼
tests/*.py  (el criterio EARS convertido en asserts ejecutables)
      │
      ▼
.github/workflows/ci.yml  (ejecuta los tests en cada push/PR, automáticamente)
      │
      ▼
✅ PR se puede mergear   ó   ❌ PR bloqueado porque rompe la spec
```

En este proyecto el harness tiene dos capas:

1. **`pytest.ini`** — configura cómo se descubren y corren los tests
   (`tests/`, con el repo en el `pythonpath`).
2. **`.github/workflows/ci.yml`** — en cada `push`/`pull request` contra
   `main` o `develop`, GitHub Actions instala `requirements.txt` y corre
   `pytest`. Si algo rompe un criterio EARS, el check en rojo lo delata
   antes de mergear.

## 4. El ciclo completo, aplicado de verdad en esta rama

Para que no quede solo en teoría, aplicamos el ciclo completo a la primera
tarea del Kanban (issue [#4](https://github.com/Bootcamp-IA-MAD-P7/proyecto9-js/issues/4),
*"Load and explore the YouTube comments dataset"*):

1. **Spec ya existía**: el criterio EARS en `spec.md` — "WHEN a raw comments
   dataset is provided, THE SYSTEM SHALL produce a cleaned, preprocessed
   dataset ready for vectorization" (la parte de carga).
2. **Task ya existía**: `tasks.md` sección 1, "Load the raw YouTube comments
   dataset into a DataFrame".
3. **Implementación** (nueva): [`src/data/loader.py`](../src/data/loader.py)
   — función `load_comments()` que lee un CSV, normaliza nombres de columna
   y valida que existan las columnas esperadas.
4. **Harness/tests** (nuevo): [`tests/test_data_loader.py`](../tests/test_data_loader.py)
   — 3 tests que convierten el criterio EARS en asserts: columnas
   consistentes, error si falta una columna requerida, y resumen de clases
   correcto.
5. **Verificación**: corrí `pytest -v` en local → los 3 tests pasan. A
   partir de esta rama, cualquier push/PR futuro los correrá automáticamente
   vía `ci.yml`.
6. **Trazabilidad**: actualicé `tasks.md` marcando esa tarea como hecha
   (`[x]`), con nota de qué falta realmente (correr `dataset_summary` contra
   el dataset real de YouTube, que aún no está en `data/raw/` — solo usé un
   CSV sintético de 4 filas para el test).

Nada de esto usa el dataset real de Google Drive del briefing todavía —
usé un CSV mínimo inventado solo para poder escribir y correr el test
sin descargar nada por ti. Ese es tu siguiente paso manual (ver sección 5).

## 5. Cómo seguir aplicando esto al resto del Kanban

Para cada issue pendiente (`Todo` o `Backlog` en el proyecto):

1. Lee el criterio EARS correspondiente en `spec.md` (si no existe todavía
   para ese nivel — medio/avanzado/experto —, se crea una nueva carpeta
   `specs/002-.../spec.md` con el mismo formato antes de tocar código).
2. Escribe primero el test en `tests/` que exprese ese criterio como
   asserts. Debe fallar (todavía no hay implementación).
3. Escribe el código mínimo en `src/` que hace pasar ese test.
4. Corre `pytest` en local. Cuando pase, haz commit y push — el harness de
   CI lo revalida automáticamente en el PR.
5. Marca la tarea como `[x]` en `tasks.md` y mueve la tarjeta del Kanban a
   `Done` (o `In Progress` si es parcial, como hicimos aquí).

### Paso manual pendiente: dataset real

Para continuar con datos reales (no el CSV sintético):

1. Descarga el dataset del enlace del briefing (Google Drive).
2. Colócalo en `data/raw/` (esa carpeta está en `.gitignore`, así que no se
   sube al repo — correcto, son datos, no código).
3. Actualiza/crea un script o notebook que llame a
   `load_comments("data/raw/<archivo>.csv")` y `dataset_summary(df)` para
   completar la exploración real (issue #4 al 100%).
