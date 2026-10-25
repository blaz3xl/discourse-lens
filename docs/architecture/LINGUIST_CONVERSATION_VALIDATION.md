# Validación de la conversación lingüística

Fecha: 25 de octubre de 2026.

**Alcance revisado tras la evaluación de Teun van Dijk:** los resultados de esta
primera ejecución acreditan los comportamientos conversacionales descritos, pero
no bastan para acreditar calidad del análisis ideológico. La respuesta que él
evaluó era vaga, se concentraba en la introducción y eludía clasificar las
orientaciones del programa. Su observación motiva el
[plan de corrección y contextualización](CONTEXTUAL_ANALYSIS_PLAN.md) y una nueva
validación; este informe no debe presentarse como aprobación experta del producto.
Los fallos de pruebas consignados en esta primera ejecución se corrigieron en la
[validación posterior](CONTEXTUAL_ANALYSIS_VALIDATION.md), que ejecuta la batería
completa. Las cifras históricas siguientes describen aquella ejecución inicial.

Se ejecutó el [plan](LINGUIST_CONVERSATION_PLAN.md) y se probó la aplicación web
con Playwright, la cuenta existente, su modelo guardado `gpt-6-luna` y la revisión
ya cargada de `Programa_Jose_Antonio_Kast_R.pdf`. No se simularon respuestas del
modelo ni de las herramientas en la prueba de navegador. Se conservó el nivel
de razonamiento configurado por el usuario.

## Resultado observado

La conversación permite pasar de una lectura para principiantes a una discusión
especializada sin elegir manualmente un marco. En este escenario el asistente
explica decisiones lingüísticas, discute una objeción, usa el corpus para contar,
rechaza una atribución falsa y ayuda a redactar sin presentar su redacción como
texto original. La prueba acredita esos comportamientos concretos; no establece
una equivalencia universal con un lingüista humano ni evalúa todo el documento.

| Intercambio real | Resultado | Tiempo | Operaciones |
| --- | --- | --- | --- |
| Lectura guiada sin conocimientos previos | Lee la página 2 y explica la repetición de «emergencia» y la construcción de un diagnóstico común. Propone una continuación concreta. | 42 s | Lectura de página |
| Seguimiento conceptual | Distingue valoración negativa de presuposición mediante «ha dejado de», conservando el tema y la localización. | 42 s | Lectura de página |
| Objeción especializada sobre Van Dijk | Distingue evaluación, pertenencia grupal e hipótesis de polarización. Discute una lectura alternativa y reconoce la necesidad de comparación. | 57 s | Lectura de página |
| Recuento de «libertad» | Obtiene 21 ocurrencias en una población de 13.928 tokens y separa frecuencia de interpretación ideológica. Recuento comprobado directamente contra el texto canónico. | 36 s | Evidencia de frecuencia |
| Verificación de una cita atribuida por otro lector | Busca literalmente «Todo inmigrante es un criminal», obtiene cero coincidencias y dice que no debe citarse como frase del PDF. Distingue la búsqueda de una interpretación más amplia. | 58 s | Búsqueda exacta |
| Ayuda para redactar | Redacta un comentario con observación, interpretación y alternativa; lo identifica como redacción propia. | 21 s | Reutiliza evidencia anterior |
| Criterio de investigación | Desarrolla la «prueba del puente» como criterio provisional e identifica una debilidad de exigir valoraciones positivas explícitas. | 73 s | Discusión conceptual |
| Memoria después del sexto intercambio | Recupera el nombre y el criterio del intercambio séptimo; lo aplica al ejemplo inicial sin dar por demostrada la polarización. | 78 s | Lectura de página |

Los ocho intercambios utilizaron seis operaciones de investigación en total.
La conversación quedó guardada y se reabrió desde el navegador.

## Cambios que sostienen el resultado

- Política explícita de interlocución: adaptación de profundidad, respuesta a
  objeciones, explicaciones con ejemplos, lecturas alternativas y pasos siguientes.
- Catálogo de herramientas con nombres, tipos, valores por defecto, restricciones
  y propiedades anidadas conservados.
- Argumentos estructurados como valores JSON nativos. La prueba real descubrió
  que codificarlos como otro JSON dentro de una cadena producía escapes inválidos.
  También se excluyen los mensajes intermedios de la salida final de Responses.
  La distinción de fases y el esquema recursivo se contrastaron con la
  [documentación oficial de Responses](https://developers.openai.com/api/docs/guides/reasoning#phase-parameter)
  y de [salidas estructuradas](https://developers.openai.com/api/docs/guides/structured-outputs#recursive-schemas-are-supported).
- Memoria con los doce mensajes más recientes, en orden, y un presupuesto de
  32.000 caracteres. La interpretación conserva prioridad sobre los datos técnicos.
- Mapa de fuentes autorizadas, referencias canónicas para lecturas directas y
  comprobación del alcance de identificadores de párrafo y pasaje.
- Entradas de lectura guiada, discusión y patrones; pasajes consultados plegables
  y énfasis tipográfico renderizado mediante nodos de texto, sin interpretar HTML.

## Comprobaciones automatizadas

- **109 pruebas aprobadas** sobre conversación, referencias, aislamiento de
  fuentes y transporte del proveedor, incluidas 16 regresiones nuevas.
- Batería adicional: **100 aprobadas y siete fallos previos**. Los mismos siete
  fallos se reprodujeron con el código original de `HEAD`, antes de los cambios.
  Corresponden a expectativas antiguas sobre atajos de teclado, campos de presets,
  descubrimiento y preferencias de modelos, consentimiento de embeddings y el
  orden de validación de índices híbridos. No se presentan como pruebas aprobadas.
- Ruff y comprobación de tipos de los archivos de conversación: sin errores.
- Playwright: ocho respuestas reales completas; uso del motor para el recuento;
  memoria del criterio del séptimo intercambio; reapertura; controles de pasajes;
  escritorio de 1440 px y móvil de 390 px; sin errores de JavaScript ni
  desbordamiento horizontal del documento.
- Snyk excluido de la validación final a petición del usuario.

## Límites y observaciones

Los tiempos observados van de 21 a 78 segundos con la configuración existente.
No se modificó el modelo ni su preferencia de razonamiento para acelerar la prueba.
El historial completo queda almacenado; el contexto inmediato del modelo sigue
siendo una ventana acotada, no una memoria semántica permanente.

La extracción del PDF contiene palabras partidas por guiones de final de línea.
En la transcripción algunas citas largas unen esos cortes y algunas menciones
metalingüísticas cambian la mayúscula inicial. Para una cita académica diplomática
hay que volver al pasaje original conservado. Los ejemplos de polarización siguen
siendo lecturas locales: esta prueba no valida un patrón en todo el programa ni
verifica sus afirmaciones históricas o estadísticas.

## Evidencia y reproducción

Los artefactos privados de la ejecución están en `output/linguist-acceptance/`,
excluido de Git: `conversation.json`, `source-verification.json`, capturas de
escritorio y móvil, lectura inicial de referencia y registros de pruebas.
No contienen la contraseña ni el estado de sesión del navegador.

El ejecutor reproducible es `scripts/verify_linguist_playwright.cjs`. Recibe la URL
y las credenciales exclusivamente a través de variables de entorno:

```bash
DL_TEST_BASE_URL="https://tu-instancia" \
DL_TEST_EMAIL="$correo_de_prueba" \
DL_TEST_PASSWORD="$clave_de_prueba" \
node scripts/verify_linguist_playwright.cjs
```

Requiere Playwright instalado. `DL_PLAYWRIGHT_MODULE` permite indicar una
instalación existente; `DL_TEST_BROWSER` permite indicar un Chromium instalado.
`DL_TEST_PROJECT` selecciona el proyecto de investigación que contiene el PDF.
`DL_TEST_STORAGE_STATE` permite reutilizar una sesión guardada fuera del repositorio.
Para confirmar únicamente la interfaz después de una ejecución completa se puede
usar `DL_TEST_RESUME_ID` con `DL_TEST_START_CASE=8`; conserva la transcripción previa.
