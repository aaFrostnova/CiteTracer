

# CITETRACER: Detección en Cascada de Alucinaciones de Citas Multiagente

[![Paper](https://img.shields.io/badge/arXiv-2605.08583-b31b1b.svg)](https://arxiv.org/abs/2605.08583) [![Dataset](https://img.shields.io/badge/HuggingFace-Hallucinated__Citation-yellow.svg)](https://huggingface.co/datasets/Afrostnova/Hallucinated_Citation)

CITETRACER detecta citas fabricadas en artículos de investigación y asigna cada cita a una de las **12 categorías de una taxonomía** (R1-R3, P1-P3, H1-H6) para que los revisores vean *qué* campo es incorrecto, no solo si la cita es falsa. La pipeline analiza entradas en PDF o BibTeX, recupera evidencia a través de una cascada de cuatro etapas (caché de memoria, recuperación por URL, ocho conectores académicos en paralelo, agente web de respaldo), ejecuta coincidencia determinística de campos y dirige los casos residuales a agentes jueces especializados por clase.

![CITETRACER overview](figs/overview.png)

En un benchmark sintético de 2.450 citas, CITETRACER alcanza un F1 por clase de 97.0 / 95.8 / 98.5 para Real / Potencial / Alucinada. En 957 citas fabricadas del mundo real procedentes de envíos rechazados en mesa de ICLR 2026 y ACM CCS 2026, detecta el 97.1 % sin abstenciones. Consulte [docs/taxonomy.md](docs/taxonomy.md) para la taxonomía completa y [docs/metric_guide.md](docs/metric_guide.md) para el protocolo de evaluación.

## Instalación

```bash
# 1) clonar
git clone https://github.com/aaFrostnova/Citation_Hallucination_Detection.git
cd Citation_Hallucination_Detection

# 2) instalar (Python 3.10+)
pip install -r requirements.txt

# 3) configurar (edite las claves descritas en "Configuración" a continuación)
cp config.example.json config.json
```


## Configuración

La pipeline lee `config.json`. Cada clave también puede ser sobrescrita por una variable de entorno del mismo nombre con el prefijo `CITATION_CHECKER_`. `config.example.json` incluye comentarios de documentación (cualquier clave cuyo nombre comience con `_` es ignorada por el cargador, así que puede dejarlos o eliminarlos libremente).

| Bloque                  | Qué controla                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `entry_extraction`      | Modelo de OCR para la detección de la región de bibliografía (solo PDF).                                                  |
| `ocr_vlm_extract`       | Agente Analizador (reanálisis VLM de bloques recortados). Establezca `provider` en una API en la nube y complete el subbloque correspondiente. |
| `verification_llm`      | Agente de Coincidencia + Jueces Especializados por Clase. `max_candidates` limita los primeros K candidatos por fuente.   |
| `connectors`            | Ruta de caché del conector, rutas de espejo de DBLP, ocho fuentes académicas, proveedor de búsqueda web y todas las claves de API relacionadas. |
| `citation_parse_method` | Fijado en `"ocr_vlm_extract"`.                                                                                             |

### Cómo completar `config.json`

Después de `cp config.example.json config.json`, edite cuatro elementos en orden:

**1. Seleccione proveedores de LLM para `ocr_vlm_extract` y `verification_llm`.**
Cada bloque tiene su propio campo `provider`. Los dos bloques son independientes, por lo que puede mezclar proveedores (p. ej., Bedrock para el Analizador, OpenAI para el Verificador).

  - `bedrock`  → establezca `bedrock.region`, `bedrock.model_id` (p. ej.
    `"qwen.qwen3-vl-235b-a22b"`) y `bedrock.bearer_token`.
  - `openai`  → establezca `bedrock.model_id` (p. ej. `"gpt-5"`) y exporte
    `OPENAI_API_KEY` en su terminal.
  - `azure_openai` → establezca `bedrock.model_id` con el nombre de su implementación de Azure
    (p. ej. `"gpt-5.4"`) y exporte
    `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`.

El nombre del campo `bedrock.model_id` es histórico; para `openai` y
`azure_openai` se interpreta como el nombre del modelo de OpenAI o el nombre
de la implementación de Azure, respectivamente.

**2. Configure `entry_extraction` si ejecutará entradas en PDF.**
Dirija `entry_extraction.local.model_path` hacia un punto de control descargado
de `DeepSeek-OCR-2`. Los usuarios solo de BibTeX (`apps.bib_checker.run`)
pueden dejar este bloque sin cambios.

**3. Seleccione un backend de Agente Web en `connectors.web_search_provider`.**
Tanto `tavily` (establezca `tavily_api_key`) como `serpapi` (establezca `serpapi_key`) son funcionalmente intercambiables.

**4. Complete las claves opcionales de los Conectores Académicos para mayores cuotas.**
Cada Conector Académico funciona sin una clave, pero `semantic_scholar_api_key`
y `ncbi_api_key` (+ `ncbi_email`) aumentan significativamente el límite de velocidad.
`openalex_mailto` coloca su tráfico en la cola de cortesía. Deje cualquier campo
vacío para omitirlo.

Después de editar, verifique que el archivo sea cargable:

```bash
python -c "from apps.pdf_checker.config import load_pdf_checker_config; cfg = load_pdf_checker_config(); print('OK', cfg.ocr_vlm_extract.provider, '/', cfg.verification_llm.provider)"
```

### Backends de LLM compatibles

`ocr_vlm_extract` (Agente Analizador) y `verification_llm` (Agente de Coincidencia +
Jueces Especializados por Clase) pasan ambos por la misma
`packages.llm.client.build_chat_client(...)` factory, por lo que la misma lista de
proveedores se aplica a ambos. La etapa de OCR (`entry_extraction`) convierte imagen a texto en la página PDF renderizada y solo se ejecuta en el paquete vLLM de DeepSeek-OCR incluido en el repositorio.

| `provider`     | Utilizado por          | Variables de entorno / config requeridas                                                                                                                                                                     |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bedrock`      | analizador, verificador| `bedrock.bearer_token`, `bedrock.region`, `bedrock.model_id`. IDs probados: `qwen.qwen3-vl-235b-a22b`, `moonshotai.kimi-k2.5`, `us.anthropic.claude-opus-4-7-20251101-v1:0`, `us.anthropic.claude-sonnet-4-5-v1:0`. |
| `openai`       | analizador, verificador| Var. env `OPENAI_API_KEY`, `bedrock.model_id` (interpretado como nombre de modelo de OpenAI; los modelos multimodales como `gpt-4o` / `gpt-5` son compatibles porque `OpenAIChatShim` traduce bloques de imagen a `image_url`). |
| `azure_openai` | analizador, verificador| Var. env `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`, `bedrock.model_id` (interpretado como nombre de implementación, p. ej. `gpt-5.4`).                                                                |
| `local`        | OCR `entry_extraction` | `entry_extraction.local.model_path` apuntando al punto de control HF de DeepSeek-OCR-2; usa el paquete multimodal vLLM incluido en `apps/pdf_checker/ingest/reference_segmenter.py`.                         |

El Agente Analizador y el Verificador solo funcionan con API en la nube. La ruta de LLM local se
eliminó porque el paquete vLLM de OCR local y un segundo LLM local suelen
competir por la memoria de la GPU y tienen requisitos incompatibles de versiones de `transformers` / `vllm`.
Los campos por bloque (`max_new_tokens`, `temperature`,
`max_candidates`) tienen efecto bajo cada proveedor compatible.

### Conectores Académicos (fuentes de datos académicos)

Cada Conector Académico consulta una fuente bibliográfica externa. La
mayoría son gratuitos sin autenticación; algunos aceptan una clave para mayores cuotas.

| Conector           | Campo en `connectors.*`                        | ¿Clave requerida?                    |
| ------------------ | ---------------------------------------------- | -------------------------------- |
| `arxiv`            | (ninguno)                                      | No.                              |
| `dblp_online`      | (ninguno)                                      | No.                              |
| `dblp_sqlite`      | `dblp_sqlite_path`                             | No (espejo sin conexión).         |
| `crossref`         | (ninguno)                                      | No.                              |
| `acl_anthology`    | (ninguno)                                      | No.                              |
| `europepmc`        | (ninguno)                                      | No.                              |
| `pubmed`           | `ncbi_api_key`, `ncbi_email`                   | Opcional. Sin una clave, NCBI limita a ~3 solicitudes/s; con una clave, ~10 solicitudes/s. |
| `semantic_scholar` | `semantic_scholar_api_key`                     | Opcional. La capa pública está severamente limitada en velocidad; una clave de API (solicitar en semanticscholar.org) elimina el límite. |
| `openalex`         | `openalex_mailto`, `openalex_api_key`          | Opcional. `mailto` lo coloca en la cola de cortesía; `api_key` es para cuota premium. |
| `url_direct`       | (usa `tavily_api_key` como respaldo)           | No, excepto para el respaldo de extracción de URL de Tavily cuando una cita solo tiene una URL vaga. |
| `web_search`       | `web_search_provider` + la clave de API correspondiente | **Sí.** Consulte la tabla de Búsqueda Web a continuación.                                            |

### Backends de Búsqueda Web (`connectors.web_search_provider`)

El Agente Web necesita un backend de búsqueda web general para completar la cascada
en citas de cola larga. Las dos opciones a continuación son funcionalmente
intercambiables: cada una devuelve los 5 mejores resultados como
`{url, title, snippet}` y el resto de la pipeline no se preocupa por cuál
los produjo. Elija aquella para la que tenga credenciales.

| `web_search_provider` | Clave(s) requerida(s)                                        | Notas                                                                                       |
| --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `tavily`              | `tavily_api_key`                                       | Predeterminado. Índice orientado a LLM de Tavily; los fragmentos son extractos en markdown del cuerpo de la página.  |
| `serpapi`             | `serpapi_key`                                          | Envuelve Google Search a través de SerpAPI; los fragmentos son los nativos de Google.                 |

Si establece `web_search_provider` en un valor cuya clave no está configurada,
la etapa del Agente Web registrará una advertencia y se omitirá (el resto de la
cascada sigue ejecutándose).

Banderas CLI útiles (`python -m apps.pdf_checker.run --help` para la lista completa):

| bandera                | predeterminado | propósito                                                  |
| ---------------------- | -------------- | -------------------------------------------------------- |
| `--paper-workers`      | 2              | artículos en paralelo cuando `--input` es un directorio    |
| `--citation-workers`   | 4              | citas en paralelo dentro de cada artículo                  |
| `--connector-workers`  | 8              | llamadas de conector en paralelo por cita                  |
| `--offline-only`       | desactivado    | omitir cada conector en línea (usar solo DBLP local)       |
| `--extract-only`       | desactivado    | ejecutar solo las etapas 1-2 y volcar las citas analizadas |
| `--resume`             | desactivado    | omitir artículos cuyo `_report.md` ya existe               |
| `--save-ocr-artifacts` | desactivado    | persistir artefactos de depuración de OCR en `<out>/ocr_artifacts/` |

## Inicio rápido

### Desde un PDF

```bash
mkdir -p artifacts/demo
python -m apps.pdf_checker.run \
  --input path/to/paper.pdf \
  --out  artifacts/demo \
  --paper-workers 1 --citation-workers 8 --connector-workers 6
```

### Desde un archivo BibTeX

```bash
python -m apps.bib_checker.run \
  --input path/to/refs.bib \
  --out  artifacts/demo \
  --paper-workers 1 --citation-workers 8 --connector-workers 6
```

Salida por archivo de entrada:

- `<stem>_report.json` — dictámenes estructurados por cita
- `<stem>_report.md`   — informe en markdown amigable para revisores
- `<stem>_report.timing.json` — latencia por fase (solo PDF)

`--input` acepta un único archivo o un directorio. `--resume` omite las entradas
cuyo `_report.md` ya existe, útil para trabajos por lotes.

## Modo solo Verificador (omitir extracción de PDF)

Si su entrada son citas ya analizadas,
omite la Etapa 1 y ejecuta solo la cascada junto con los jueces. Las entradas son archivos
JSON con el esquema producido por la Etapa 1 (`results/eval_*_rule_based.json`).

```bash
python scripts/eval_judge_only.py \
  --inputs   results/eval_*_rule_based.json \
  --out      results/judge_only_run \
  --model    qwen.qwen3-vl-235b-a22b \
  --workers  16
```

Esto también acepta `--exclude-connectors` para excluir experimentalmente cualquier grupo de conectores
(p. ej. `--exclude-connectors web_search`).

## Conjuntos de datos

Dos conjuntos de datos se incluyen con el repositorio:

| Ruta                                       | Qué es                                                                                                                | Tamaño  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ----- |
| `data/synthetic_data/v2/`                  | 2.450 citas sintéticas a través de las 12 categorías (un JSON por subtipo: `R1.json`, `R2.json`, `R3.json`, `P1.json`, `P3.json`, `H1.json`..`H6.json`; `P2` está reservado para citas no académicas y no está en esta instantánea), con `meta.json` llevando la etiqueta de referencia por cita. Construido a partir de semillas BibTeX reales con mutaciones LLM controladas. | 2.450 |
| `data/iclr2026_hallucinated/`              | 957 citas fabricadas del mundo real de envíos rechazados en mesa de ICLR 2026. `hallucinated_refs.json` es la lista cruda; `hallucinated_refs_structured.json` es la versión de registro estructurado analizado utilizada por el verificador. | 957   |

El conjunto sintético es el benchmark principal utilizado en cada tabla del paper; el
conjunto del mundo real es la prueba fuera de distribución en la Sección 4.4.

Un espejo de ambos conjuntos de datos también está publicado en Hugging Face:
[Afrostnova/Hallucinated_Citation](https://huggingface.co/datasets/Afrostnova/Hallucinated_Citation).

## Reproducir los resultados del paper

```bash
# verificación principal en el benchmark sintético de 2.450 citas
bash scripts/eval_H_rule_based.sh
bash scripts/eval_R_rule_based.sh
```

## Estructura del repositorio

```
apps/pdf_checker/         punto de entrada PDF de extremo a extremo (run.py, config.py, ingest/)
apps/bib_checker/         punto de entrada solo BibTeX (omite la Etapa 1, ejecuta el verificador directamente)
packages/connectors/      conectores bibliográficos + caché + orquestador
packages/core/            verificador, agentes en cascada, coincidentador de campos, jueces
packages/llm/             cliente LLM agnóstico al proveedor (bedrock / openai / azure / vLLM local)
scripts/                  evaluación, benchmarks, puntuación, plantillas sbatch
data/synthetic_data/v2/   benchmark sintético de 2.450 citas, un JSON por subtipo
data/iclr2026_hallucinated/  957 citas fabricadas del mundo real de rechazos de mesa de ICLR 2026
docs/                     taxonomía, guía de métricas, notas de omisión de renderizado
tests/                    pruebas unitarias con pytest para analizador, normalizador, agentes
```

## 📌Cita

Si considera útil este repositorio, por favor cite el paper:

```bibtex
@article{li2026source,
  title={Source or It Didn't Happen: A Multi-Agent Framework for Citation Hallucination Detection},
  author={Li, Mingzhe and Lin, Zhiqiang and Ma, Shiqing},
  journal={arXiv preprint arXiv:2605.08583},
  year={2026}
}
```
