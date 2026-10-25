# DiscourseLens: conversar con un lingüista sobre documentos

Estado: ejecutado. Véase la [validación con el modelo real](LINGUIST_CONVERSATION_VALIDATION.md).

## Objetivo

Convertir el chat en un interlocutor que ayuda a leer, aprender, investigar y
discutir un documento. Debe adaptar la profundidad a la conversación, conservar
el hilo y justificar sus interpretaciones con pasajes comprobables. No basta con
nombrar autores o producir informes: tiene que responder al argumento concreto
del usuario y ayudarle a avanzar.

## Diagnóstico inicial

- El resumen del catálogo MCP borra las propiedades de los argumentos: el modelo
  recibe herramientas cuyos parámetros no puede conocer correctamente.
- La memoria consulta los primeros doce mensajes, por lo que después de seis
  intercambios deja de incorporar la conversación reciente.
- La memoria recorta por el final y puede conservar resultados técnicos mientras
  elimina la interpretación a la que responde el usuario.
- Cada turno empieza sin una orientación sobre las fuentes adjuntas y sin una
  distinción clara entre explicar, explorar, contrastar, contar y ayudar a escribir.
- La bienvenida ofrece una caja de texto, pero poca ayuda para quien no sabe
  formular una pregunta de investigación.

## Plan de ejecución

1. **Herramientas utilizables.** Conservar nombres, tipos, límites y valores por
   defecto de sus argumentos, incluidos objetos anidados. Mantener las
   operaciones del motor y la elección contextual de métodos por el modelo.
2. **Un interlocutor con criterio.** Incorporar un contrato de conversación:
   respuesta directa; adaptación gradual al conocimiento del usuario;
   explicaciones con ejemplos; distinción entre observación e inferencia;
   desacuerdo razonado; hipótesis alternativas y una continuación útil.
   Integrar pragmática, semántica, gramática, retórica, argumentación y análisis
   del discurso sin fingir que un recuento valida una interpretación.
3. **Lectura situada.** Dar al modelo un mapa acotado de las fuentes autorizadas.
   Leer y buscar antes de afirmar algo sobre ellas; combinar los métodos cuando
   la pregunta lo requiera. No inventar citas, contexto histórico ni estados
   mentales; no convertir una búsqueda vacía en una afirmación de ausencia.
4. **Continuidad real.** Recuperar los mensajes más recientes en orden, preservar
   primero el contenido de la discusión y separar citas/resultados de la voz
   del interlocutor. Mantener datos internos necesarios para volver al pasaje.
5. **Ayuda para empezar.** Ofrecer entradas concretas para lectura guiada,
   discusión de una hipótesis y exploración de patrones. El usuario puede
   escribir libremente y cambiar de profundidad durante la conversación.
6. **Verificación y corrección.** Añadir regresiones sobre herramientas, memoria,
   aislamiento documental y citas. Verificar tipos y calidad estática. Snyk queda
   excluido de la validación final por indicación posterior del usuario.
   Probar con Playwright y el modelo real de la cuenta existente el PDF de Kast.

## Criterios de aceptación

- Un principiante recibe un ejemplo del PDF, una explicación comprensible y una
  propuesta concreta para continuar, sin necesitar seleccionar un marco teórico.
- Una objeción especializada recibe una respuesta al argumento, distinciones
  conceptuales pertinentes y límites de la inferencia, sin asentimiento automático.
- Las afirmaciones documentales se apoyan en texto recuperado de las fuentes
  autorizadas; las citas corresponden al original y localizan el pasaje.
- Las preguntas de seguimiento y los cambios de criterio posteriores al sexto
  intercambio llegan al modelo con sus antecedentes recientes.
- Los recuentos usan operaciones del motor y explican población/alcance; las
  interpretaciones y propuestas de reescritura se distinguen del texto original.
- La interfaz permite iniciar y reabrir la conversación; funciona en escritorio
  y móvil, sin errores de JavaScript ni desbordamiento horizontal.

## Evidencia de validación

Se conservarán los resultados automatizados y las transcripciones de la prueba
real en `output/linguist-acceptance/` (excluido de Git). El informe de validación
documentará resultados observados y limitaciones, sin credenciales.
