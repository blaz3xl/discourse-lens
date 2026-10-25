# Saldo estimado y consumo

El importe pequeño junto a los ajustes del chat permite controlar un saldo
introducido por la persona que usa la cuenta. Al pulsarlo se abre «Saldo y
consumo» dentro de la configuración existente.

1. Revisa y guarda las tarifas del modelo de chat, en unidades monetarias por
   millón de tokens de entrada y salida. Para `gpt-6-luna` se ofrece una
   sugerencia de tarifas estándar en USD, consultada el 25/10/2026; no se instala
   automáticamente ni se actualiza por su cuenta. Contrástala con las condiciones
   de tu proveedor.
2. Introduce el importe disponible que quieres controlar y su moneda. «Establecer
   saldo» toma ese momento como punto de partida: no vuelve a descontar el
   historial anterior. Establecer otro importe inicia otro cómputo.
3. Cada llamada terminada que aporta uso confirmado y una tarifa guardada reduce
   el restante estimado. Una pregunta puede generar varias llamadas al modelo y
   una búsqueda de contexto. Durante la respuesta se consulta el registro cada
   cuatro segundos; también se actualiza al terminar, al recuperar el foco y
   periódicamente mientras la página está visible. Consultar el saldo no llama
   al modelo ni vuelve a cobrar una ejecución.

El símbolo `≈` identifica una estimación. No hay una cartera ni un sistema de
pagos de DiscourseLens detrás de ese número. No consulta automáticamente el
saldo prepago de OpenAI u otro proveedor, no añade fondos y no impone un límite
de gasto. El importe inicial lo debe proporcionar el usuario; no se deduce de
los tokens ni se inventa a partir del historial.

## Qué entra en el cálculo

Se cuentan las ejecuciones de modelos de texto registradas en el espacio de
esa cuenta, incluidas las llamadas de planificación y las respuestas. El
consumo de entrada y salida procede de la respuesta del proveedor, no de una
estimación del número de palabras visibles. El transporte de OpenAI conserva
también el uso del evento final de streaming, tanto en Responses como en Chat
Completions; este último solicita el bloque de uso cuando se utiliza el
adaptador oficial de OpenAI. Los adaptadores que no entreguen uso al registro
dejarán el coste como desconocido.

Las tarifas son versiones inmutables: cada ejecución conserva la que tenía al
empezar. Cambiar la tarifa no reescribe las llamadas anteriores. Los tokens de
razonamiento se incluyen una sola vez dentro de la salida total, salvo que una
tarifa explícita del motor desglosara esa categoría.

La búsqueda web nativa registra cuántas llamadas devolvió el proveedor. El
presupuesto guarda una tarifa de 0,01 USD por llamada, revisada el 25/10/2026,
y añade ese coste al de tokens. La sugerencia de tarifas del modelo y el precio
de búsqueda se basan en la [ficha del modelo](https://developers.openai.com/api/docs/models/gpt-6-luna)
y en la [tabla de precios](https://developers.openai.com/api/docs/pricing).
La descarga directa de una fuente no añade por sí sola una llamada de modelo.

No se incluyen embeddings, impuestos, cargos o actividad de otras aplicaciones,
ni diferencias de tarifa por caché, contexto largo, modalidad de servicio o
descuentos. Por eso la cifra no sustituye la factura ni el panel del proveedor.
La [documentación del uso en streaming](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
advierte que una interrupción puede impedir recibir el uso final: la ausencia
de ese dato no prueba un coste cero.

Si falta uso o tarifa, el restante pasa a «Saldo pendiente» y se muestra el
gasto conocido y los registros que faltan. Tampoco se suman divisas diferentes:
una búsqueda cobrada en USD hace incompleto un presupuesto en otra moneda.
Los importes se calculan con decimales y se guardan como texto decimal. La
pantalla redondea hasta cuatro decimales para que los consumos pequeños sean
visibles. Un restante negativo permanece visible; no bloquea solicitudes.

## Persistencia, privacidad y pruebas

El presupuesto queda en los metadatos del espacio de la cuenta. No se comparte
con las demás cuentas ni se envía al proveedor. «Quitar saldo» elimina esa
configuración y conserva el historial de consumo y las tarifas. La API de
consulta abre el proyecto en modo de solo lectura y sigue disponible mientras
otra conexión guarda evidencia.

Las rutas autenticadas son `GET /api/spending`,
`POST /api/spending/budget`, `DELETE /api/spending/budget` y
`POST /api/spending/price`. Las modificaciones con sesión de navegador exigen
el mismo origen. No se exponen credenciales en estos resultados.

Las pruebas cubren precisión, tarifas históricas, costes desconocidos, monedas,
reinicio del cómputo, lectura concurrente e independencia entre cuentas.
`scripts/verify_spending_playwright.cjs` comprueba la interfaz con una cuenta y
un modelo reales. Exige indicar un presupuesto temporal de prueba, rechaza
reemplazar uno existente y lo quita al terminar. Los registros de la llamada
real quedan disponibles para inspección. Los resultados observados se recogen
en el [informe de validación](architecture/CONTEXTUAL_ANALYSIS_VALIDATION.md).
