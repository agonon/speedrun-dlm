

# speedrun-dlm: Un benchmark de speedrun para modelos de lenguaje de difusión

**Entrena un modelo de lenguaje de difusión (DLM) lo más rápido posible para generar texto decente.**

Este repositorio investiga qué tan rápido se puede entrenar un modelo de lenguaje de difusión en *FineWeb*, utilizando *8 H100*, hasta que genere muestras de 1024 tokens que sean lo suficientemente buenas para pasar un control de calidad sencillo.

**El tablero de clasificación se basa en las muestras generadas que superan este control (usando PPL de generación más comprobaciones adicionales para los modos de fallo del PPL por sí solo, como la repetición de tokens y la coherencia).**

El **control** verifica:
- una repetición excesiva,
- coherencia básica del documento,
- calidad según un LLM de referencia fijo (`Qwen2.5-7B`),
- otras heurísticas simples (más detalles a continuación).

El proyecto está inspirado en el speedrun **[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) de Keller Jordan y colaboradores, adaptado a modelos de lenguaje de difusión**, donde muchas decisiones de diseño fundamentales aún están abiertas.

Fijamos un backbone DDiT pequeño (170M de parámetros) e incluimos reimplementaciones de **7 objetivos de difusión de bloque único comunes** en `speedrun_dlm/train_dlm.py`.

> Nota. Los métodos e ideas originales pertenecen a sus respectivos autores. Este es un proyecto independiente y no está afiliado con modded-nanogpt, sus colaboradores, ni con los autores de los artículos que citamos.


## Contenido

- [Tablero de clasificación](#leaderboard)
- [Reglas](#rules)
- [Modelo](#model)
- [Control de calidad](#quality-gate)
- [Cómo enviar](#how-to-submit)
- [Ejecutar](#run)
- [Verificar un candidato](#check-a-candidate)
- [Apéndice](APPENDIX.md)
- [Agradecimientos](#acknowledgements)

## Tablero de clasificación

**Las ejecuciones se clasifican por tiempo de entrenamiento para superar el control de calidad de las muestras.**

Para aparecer en el tablero de clasificación, un entrenador+muestreador DLM debe:
- superar al menos 42 de 50 semillas, y
- costar menos de 95.86 TFLOPs para una muestra de 1024 tokens (este umbral se explica en el [apéndice](APPENDIX.md)).

Se aceptan cambios pequeños de ajuste, nuevos muestreadores y nuevos objetivos.

El límite de FLOPs está diseñado para mantener los muestreadores razonables en términos de costo de inferencia. De lo contrario, un muestreador de difusión podría intentar pasar el control usando un número arbitrariamente grande de pasos de eliminación de ruido.

| Clasificación por tiempo de entrenamiento | Entrada | Tiempo de entrenamiento promedio (para clasificación) | Resultado del control (necesario 42/50+) | Costo de generación de 1024 tokens (necesario <95.86 TFLOPs) | Llamadas al modelo para 1024 tokens | Tokens de entrenamiento | Pasos de entrenamiento | Registro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `subs_mask` | [27m45.2s](records/0001-subs-mask-s344/training_runs.jsonl) | [44/50 seeds](records/0001-subs-mask-s344/gate_passes.tsv) | [95.16 TFLOPs](records/0001-subs-mask-s344/auxiliary_metrics.json) | [344](records/0001-subs-mask-s344/auxiliary_metrics.json) | [3.28B](records/0001-subs-mask-s344/training_runs.jsonl) | [8000](records/0001-subs-mask-s344/training_runs.jsonl) | [0001](records/0001-subs-mask-s344/) |

Los TFLOPs de inferencia son los FLOPs contados por PyTorch durante el muestreo, más los FLOPs de atención que PyTorch no contó. Consulte el [apéndice](APPENDIX.md) para la fórmula.

## Reglas

- Mismos *datos*: datos tokenizados GPT-2 de FineWeb fijos.
- Mismo *evaluador*: `Qwen/Qwen2.5-7B` fijo.
- Mismo *hardware*: 8 GPUs H100 de 80GB.
- Mismo *control de calidad*: 128 muestras incondicionales de 1024 tokens.
- Una semilla supera el control si al menos 108 de 128 muestras lo superan.
- Un muestreador DLM debe costar menos de 95.86 TFLOPs para una muestra de 1024 tokens.
- Las ejecuciones se clasifican por el tiempo promedio de entrenamiento sobre 50 semillas.
- Un récord requiere al menos 42 de 50 semillas para superar el control.

Si bien se aceptan modificaciones arquitectónicas, la cantidad de parámetros del backbone DDiT debe permanecer comparable.

## Modelo

El benchmark fija un backbone DDiT pequeño a escala aproximada de nanoGPT-2:

- 12 capas,
- ancho 768,
- 12 cabezas,
- longitud de contexto 1024,
- tokenizador GPT-2.

Los cambios específicos de difusión son:

- atención no causal,
- condicionamiento por tiempo de eliminación de ruido a través de normalización de capa adaptativa,
- incrustación de entrada y capa de salida desacopladas.


Consulte el [apéndice](APPENDIX.md) para la breve comparación con la referencia AR utilizada para establecer el límite de FLOPs.

## Control de calidad

**Los DLMs no son fáciles de comparar** porque:
- no hay acceso directo a la **probabilidad a nivel de secuencia** asignada por un modelo: a diferencia de los modelos AR, esto requiere estimar la suma sobre todos los caminos posibles de eliminación de ruido para alcanzar una muestra dada, lo cual es costoso y conduce a estimaciones de alta varianza,
- las **diferentes funciones de pérdida** en las recetas populares actuales no son directamente comparables (incluso _si_ constituyen un límite variacional de la log-verosimilitud negativa, estos límites pueden tener diferente rigidez entre parametrizaciones),
- el **muestreador** utilizado en el momento de la inferencia debe tenerse en cuenta, ya que el mismo checkpoint DLM puede producir texto diferente dependiendo del procedimiento de muestreo que genera las secuencias.

Este benchmark **evalúa un modelo entrenado junto con un muestreador** mediante:
<ol>
  <li> generando 128 muestras de 1024 tokens cada una (partiendo de una secuencia completamente ruidosa/enmascarada para DLMs), y </li>
  <li> verificando cuántas de las muestras superan una regla de calidad diseñada cuidadosamente (repetición, fluidez, etc.). </li>
</ol>

Nos referimos a [`score_generation_quality.py`](speedrun_dlm/score_generation_quality.py) y [`generation_quality_rule.json`](generation_quality_rule.json) para los detalles.

## Cómo enviar

Para incluir tu método en el tablero de clasificación, **abre un pull request** con:

- una nueva carpeta bajo `records/`,
- una nueva fila en [`records/records.csv`](records/records.csv),
- cualquier código necesario para ejecutar el método.

Si usas las salidas del entrenador, muestreador y evaluador actuales, puedes crear la carpeta del registro con [`records/make_record.py`](records/make_record.py):

```bash
python records/capture_environment.py > results/my_method/environment.json

python records/make_record.py \
  --record-dir records/nextfreeid-my-method \
  --record-id nextfreeid \
  --entry my_method \
  --trainer dlm \
  --sampler S=mysamplingsteps \
  --training-runs-jsonl results/my_method/training_runs.jsonl \
  --gate-passes-tsv results/my_method/gate_passes.tsv \
  --inference-cost-json results/my_method/inference_cost.json \
  --environment-json results/my_method/environment.json \
  --min-passes 42 \
  --records-csv records/records.csv
```

Cada entrada enviada debe incluir 50 semillas, informar cuántas semillas superan el control, conservar una instantánea de `environment.json` y mantenerse por debajo de 95.86 TFLOPs para una muestra de 1024 tokens. Verificaremos el tiempo y los artefactos del registro en nuestro clúster para cada PR.

## Ejecutar

![Edit, run, check, record](assets/demo.gif)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash prepare_data.sh
bash run_ar.sh
bash run_dlm.sh
```

## Verificar un candidato

```bash
python -m speedrun_dlm.score_generation_quality path/to/checkpoint.pt \
  --samples 128 \
  --tokens 1024 \
  --top_k 0 \
  --num_sampling_steps 344 \
  --sampling_eps 1e-3 \
  --require_significance \
  --output_dir results/gate
```

```bash
python -m speedrun_dlm.measure_inference_cost path/to/checkpoint.pt \
  --tokens 1024 \
  --top_k 0 \
  --num_sampling_steps 344 \
  --sampling_eps 1e-3 \
  --output_json results/inference_cost.json
```

## Apéndice

**Más detalles sobre objetivos, muestreadores, arquitectura y figuras se encuentran en [el apéndice](APPENDIX.md).**

## Agradecimientos

Este repositorio se basa en mucho trabajo público útil. En particular, utilizamos:

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) para la configuración del speedrun y la referencia AR,
- [MDLM](https://github.com/kuleshov-group/mdlm), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion), [D3PM](https://github.com/google-research/google-research/tree/master/d3pm) y [DUO](https://github.com/s-sahoo/duo) para los diferentes objetivos y muestreadores DLM,
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2) y [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) para los datos y el evaluador de PPL de generación.

## Cómo citar

La primera versión de este benchmark fue creada por Antoine Gonon, Adrian Müller, Léon Zheng, Zebang Shen, Clément Lalanne, Ya-Ping Hsieh, Anthony Bardou y Nicolas Boumal.

```bibtex
@misc{speedrun_dlm,
  title = {speedrun-dlm: A speedrun benchmark for diffusion language models},
  author = {Gonon, Antoine and Müller, Adrian and Zheng, Léon and Shen, Zebang and Lalanne, Clément and Hsieh, Ya-Ping and Bardou, Anthony and Boumal, Nicolas},
  year = {2026}
}
```

## Contacto

Para preguntas privadas o comentarios, no dudes en contactar a `antoine.gonon@epfl.ch`.

Para errores, problemas de reproducibilidad o cambios propuestos, por favor abre un issue o PR en GitHub.
