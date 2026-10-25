# Validación de análisis contextual y conversación

Fecha: 25 de octubre de 2026. Se ejecutó el
[plan de corrección](CONTEXTUAL_ANALYSIS_PLAN.md) sobre el PDF ya incorporado de
Kast, con la cuenta de evaluación y su modelo guardado `gpt-6-luna`. No se
simularon respuestas de modelo ni resultados HTTP en las pruebas Playwright.
No se cambió el modelo ni la preferencia de razonamiento de la cuenta.

## Qué se probó

La pregunta original se formuló antes de sugerir categorías. La revisión final
consultó las 39 páginas, una biografía de la Biblioteca del Congreso Nacional,
los estatutos del partido y una noticia contemporánea sobre el lanzamiento de
campaña. Las fuentes y sus fragmentos están guardados en el proyecto. Consultar
un antecedente no significa que se haya verificado independientemente cada
afirmación de ese sitio ni que se haya reconstruido todo el contexto chileno.

| Intercambio | Observación de contenido | Tiempo observado |
| --- | --- | --- |
| Pregunta original, sin etiquetas sugeridas | Formula una configuración nacional-conservadora y económica neoliberal; conecta iniciativa privada, impuestos, gasto, familia, fronteras y coerción con propuestas de páginas posteriores. La consulta inicial se repitió después de afinar la investigación de antecedentes. | 476 s; repetición 253 s |
| Actores, beneficios y cargas | Distingue gobierno, Estado existente, empresas, trabajadores, familias, fuerzas de seguridad y migrantes; separa lo que promete el programa de sus efectos reales. | 233 s |
| Objeción sobre nacionalismo, catolicismo, racismo y neoliberalismo | Define criterios, defiende las orientaciones nacional y económica, busca principios institucionales y la historia del partido, matiza la tradición moral y argumenta qué falta para establecer racialización. No se exigió aceptar todas las etiquetas del investigador. | 498 s |
| Teoría, contexto y tiempo | Distingue contexto histórico, representación textual, modelos de situación y de contexto, ideologías y actitudes. Separa la fecha interna de la difusión y el libro de 2024 del documento de 2025. | 332 s |
| Primera guía para principiantes | Da una observación y un paso siguiente, pero desplaza imprecisamente el referente de «una emergencia en seguridad» hacia «la seguridad». Este caso no se considera suficiente por haber pasado las comprobaciones de transporte. | 72 s |
| Revisión de la paráfrasis | Vuelve a la página 2, identifica «emergencia» como núcleo del antecedente y corrige explícitamente su explicación. | 62 s |
| Nueva guía tras reforzar fidelidad de paráfrasis | Usa un ejemplo de la página 4, explica la agencia del gobierno, el mérito y el favoritismo, y propone contrastar una medida de las páginas 26–27. | 88 s |

Se completaron seis intercambios en la conversación extensa y dos en la
comprobación final de contexto y guía. Las consultas globales iniciales tardaron
aproximadamente cuatro a ocho minutos con esta configuración. Es una limitación
práctica; no se presenta como una conversación de respuesta instantánea.

## Evidencia comprobada

- Documento: `Programa_Jose_Antonio_Kast_R.pdf`, 39 páginas y 89.464 caracteres
  canónicos. SHA-256 del original:
  `ab388567fdb4bd99a09dd9d63aaa4eae190a86103f0aec2ca23adbafaab7f707`.
- La huella del texto canónico sigue siendo
  `ff77752b3a9d2e5d145462d7f83aed20c0bb5991344b201e1aad63db2323bd5d`.
  Cinco lecturas continuas se cotejaron por coordenadas contra ese texto. Se
  comprobaron también los bytes del PDF original y la integridad de SQLite.
- Ocho fragmentos web guardados coinciden con los resultados de descarga en URL,
  extracto y huella. La huella identifica el contenido descargado; no representa
  una verificación de la verdad de sus afirmaciones.
- Se reabrieron conversaciones y fuentes desde el navegador. En escritorio de
  1440 px y móvil de 390 px no hubo errores JavaScript ni desbordamiento
  horizontal del documento. Se inspeccionaron ambas capturas finales.
- La consulta de mensajes se hizo de solo lectura para que guardar evidencia no
  bloqueara las actualizaciones. Una regresión autenticada comprueba la lectura
  con otra conexión manteniendo una transacción de escritura.
- Se normalizan enlaces a nombres de archivos únicamente cuando sus páginas
  figuran entre las inspeccionadas. No se inventan URLs de descarga ni se
  convierten páginas desconocidas en referencias verificadas.

Los primeros ensayos públicos detectaron una omisión de contexto inicial y un
bloqueo de lectura durante operaciones de evidencia. Se corrigieron. Después,
la conexión pública se volvió intermitente desde el entorno de prueba; la
validación completa terminó en contenedores locales con la misma imagen y el
mismo espacio de cuenta. Se conservan las ejecuciones públicas fallidas. No se
presenta esta ejecución como una prueba continua de disponibilidad del dominio.
La imagen final se instaló también en el servicio web existente.

## Pruebas del código

- **Batería completa final: 3.921 pruebas aprobadas, seis omitidas y ningún fallo**,
  en 264,82 segundos. Incluye lectura, herramientas, contexto, citas,
  conversación, transporte del proveedor y contratos de solo lectura/red.
- Ruff y comprobación de tipos del proyecto: sin errores. Tras el ajuste visual
  del saldo se aprobaron también las 81 pruebas de web, contexto y presupuesto,
  y la comprobación de sintaxis del JavaScript de la página.
- Se corrigió la validación de identificadores híbridos antes de consultar el
  preset y se exigió autorización explícita para enviar datos a embeddings.
  También se actualizaron contratos de pruebas que todavía esperaban campos,
  versiones metodológicas, argumentos o herramientas anteriores.
- Una prueba conserva la procedencia inicial tras doce intercambios adicionales
  y la recupera en modo de solo lectura y sin conexión. La recuperación usa los
  artefactos del motor, incluso si los metadatos copiados por un mensaje difieren.
- Según las instrucciones `AGENTS.md` entregadas al continuar el trabajo, se
  ejecutó Snyk sobre código de aplicación y ejecutores. Los ejecutores no tienen
  avisos. En aplicación aparecen tres avisos también presentes en el commit
  base: una ruta de configuración tomada del entorno y dos flujos hasta una
  consulta que construye únicamente marcadores SQL y pasa los valores como
  parámetros. La comparación no encontró avisos nuevos. No se interpreta este
  resultado como una garantía general de seguridad.

## Saldo junto a los ajustes

Se añadió un presupuesto manual por cuenta con tarifas editables, conservación
de la tarifa de cada ejecución y actualización desde registros confirmados.
No se implementó una consulta al crédito real del proveedor. El indicador usa
`≈`, deja el restante sin calcular cuando faltan costes y no convierte divisas.
La [guía de saldo](../SPENDING.es.md) describe su alcance.

La primera prueba Playwright detectó que el streaming no guardaba los tokens del
evento final, aunque la respuesta se completara correctamente. Se corrigió en
el transporte de OpenAI y se añadieron regresiones que atraviesan transporte,
registro y presupuesto, tanto para salida normal como estructurada. Los
adaptadores que no proporcionen uso mantienen un coste desconocido.

La repetición usó la cuenta existente, su modelo real y la conversación guardada,
sin respuestas simuladas. Un presupuesto temporal de **1 USD** pasó a
**0,9973299 USD**, con 23.456 tokens de entrada y 649 de salida, de los que 516
eran de razonamiento. El coste estimado fue **0,0026701 USD** y se mostró como
`≈ 0,9973 US$`. Una nueva consulta al registro devolvió el mismo importe sin
volver a descontarlo. La respuesta terminó en aproximadamente diez segundos.

Se comprobó la colocación junto al ajuste del modelo en escritorio y móvil,
sin errores JavaScript. La inspección visual detectó un mínimo de anchura
heredado que recortaba el campo de moneda; se acotó el campo a su columna y se
ajustó la apertura del panel. La confirmación final en ambas anchuras mostró
el título visible y los campos dentro del panel, sin otra llamada de modelo.
También se comprobó el botón «Quitar saldo». El presupuesto temporal se quitó al terminar;
las tarifas y el registro de la ejecución real se conservaron. No se dejó una
cantidad inventada como saldo de la cuenta.

## Alcance disciplinar

La revisión metodológica utilizó el material aportado por el investigador y los
apartados teóricos y el caso chileno de *Discourse and Ideologies of the Radical
Right* (2024). La transcripción aportada contiene intervenciones del autor y de
ChatGPT; las respuestas de ChatGPT no se adoptaron como fuentes históricas.

La metodología final es `builtin.van-dijk-ideology` 1.7.0 y
`builtin.sociocognitive` 1.4.0. La conversación extensa comenzó con la versión
ideológica 1.6.0; se repitió la pregunta inicial con 1.7.0 y la investigación de
antecedentes reforzada. La guía final se probó después de añadir la instrucción
de comprobar referentes y polaridad en la política de conversación.

La nueva respuesta ofrece más que una lista de temas y no convierte cualquier
intervención estatal en una refutación del neoliberalismo. Sin embargo, siguen
siendo interpretaciones producidas por un modelo, con selecciones de contexto
acotadas. Se observó una equivocación de paráfrasis y variación en la forma de
citar; las citas extensas pueden normalizar cortes de línea de la extracción.
No se acredita una reconstrucción empírica de las creencias de un grupo ni de
las intenciones privadas del candidato. Tampoco se acredita una implementación
completa de las teorías o una equivalencia con un lingüista humano.

La valoración disciplinar de Teun van Dijk u otros especialistas sigue pendiente.
Este informe es una descripción de implementación y pruebas, no su aprobación.

## Reproducción y artefactos

`scripts/verify_linguist_playwright.cjs` usa Playwright y un proveedor real. Sus
variables principales son `DL_TEST_BASE_URL`, credenciales mediante
`DL_TEST_EMAIL` / `DL_TEST_PASSWORD` o un estado de sesión privado,
`DL_TEST_PROJECT`, `DL_TEST_SCENARIO=contextual` y `DL_TEST_OUTPUT`.
`DL_TEST_RESUME_ID`, `DL_TEST_START_CASE`, `DL_TEST_CASE_NAMES` y
`DL_TEST_CASE_LIMIT` permiten reabrir o repetir partes sin simular respuestas.
El ejecutor distingue sus comprobaciones técnicas de la revisión semántica.

Las transcripciones, resultados de herramientas, capturas y registros privados
se conservan en `output/contextual-analysis/`, excluido de Git. Las ejecuciones
completas están en `final-local/` y `final-profile/`; la comprobación documental
está en `source-verification.json`. La prueba de gasto está en `spending-browser/`
y el primer fallo de streaming en `spending-browser-initial/`.
`scripts/verify_spending_playwright.cjs` permite repetir la prueba de saldo con
un presupuesto temporal explícito. No se publican credenciales, sesiones, la
correspondencia aportada ni el PDF del libro.
