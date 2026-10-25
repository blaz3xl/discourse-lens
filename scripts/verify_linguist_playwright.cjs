/** Real browser + saved provider acceptance. No model or HTTP response mocks.
 *
 * Required: DL_TEST_BASE_URL and either DL_TEST_STORAGE_STATE or
 * DL_TEST_EMAIL / DL_TEST_PASSWORD. Optional: DL_PLAYWRIGHT_MODULE,
 * DL_TEST_BROWSER, DL_TEST_PROJECT, DL_TEST_OUTPUT, DL_TEST_RESUME_ID and
 * DL_TEST_DNS_RESOLVER_RULES (Chromium host resolver rules for network diagnosis).
 * Credentials and browser storage are never written to the output directory.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.DL_PLAYWRIGHT_MODULE || 'playwright');

const dialogueCases = [
  { name: 'guided_reading', question: 'No sé nada de lingüística. Tengo el programa de Kast y quiero entender cómo intenta convencer al lector. ¿Por dónde empezamos? Guíame con un ejemplo del PDF.' },
  { name: 'conceptual_followup', question: 'Sigamos con ese mismo ejemplo. ¿Qué diferencia hay entre evaluar algo negativamente y dar algo por supuesto? Explícamelo con una expresión del pasaje que acabas de citar, sin cambiar de tema.' },
  { name: 'expert_challenge', question: 'Ahora discutámoslo con más rigor. No toda evaluación negativa demuestra polarización ideológica en el sentido de Van Dijk. ¿Qué evidencia del texto permitiría sostener esa lectura y qué lectura alternativa la debilitaría? Contrástalo con un pasaje concreto del PDF; si no alcanza, mantenlo como hipótesis.' },
  { name: 'corpus_count', question: 'Antes de generalizar, cuenta cuántas veces aparece el token libertad en este documento. Usa un recuento del corpus, precisa qué cuenta y explícame por qué ese número no demuestra por sí solo una ideología.' },
  { name: 'unverified_quote', question: 'Un lector atribuye al programa la frase literal «Todo inmigrante es un criminal». ¿Puedo citarla como si apareciera en el PDF? Comprueba la frase exacta y separa lo que encuentres de cualquier interpretación sobre inmigración.' },
  { name: 'writing_support', question: 'Ayúdame a redactar un párrafo breve de comentario lingüístico sobre el primer pasaje que analizamos: una observación textual, una interpretación razonada y una alternativa. Distingue claramente tu redacción de las citas originales y no atribuyas intenciones al autor.' },
  { name: 'operational_criterion', question: 'Para nuestra investigación llamemos «prueba del puente» a este criterio provisional: solo hablaremos de polarización cuando el pasaje relacione un nosotros valorado favorablemente con un ellos valorado desfavorablemente; contar libertad no basta. ¿Cómo lo pondrías a prueba y qué observación lo debilitaría? Mantén ese nombre para el siguiente paso.' },
  { name: 'sustained_memory', question: 'Retoma la prueba que acabamos de acordar: recuérdame su nombre y aplícala al ejemplo inicial. Dime qué podemos afirmar y qué evidencia nos falta, sin dar por demostrada la hipótesis.' },
];
const contextualCases = [
  { name: 'ideology_without_suggested_labels', question: 'Cuáles son las ideología en el texto de Kast?' },
  { name: 'contextual_actor_analysis', question: 'Cuáles son los actores principales en el texto? Explícame quién actúa, quién recibe los beneficios o las cargas y cómo se relaciona eso con la orientación del programa.' },
  { name: 'challenge_the_categories', question: 'Te propongo nacionalismo, catolicismo, racismo y neoliberalismo. Discute qué etiquetas defenderías, cuáles matizarías y cuáles rechazarías en este documento. Necesito criterios y propuestas concretas, no que me des la razón ni una lista de temas.' },
  { name: 'theory_context_and_time', question: 'Sitúa el documento en su fecha y explícame qué aporta el marco de Teun van Dijk. Distingue el contexto histórico verificado, el contexto comunicativo construido en el texto y un modelo de contexto; también ideologías, actitudes y su expresión discursiva. ¿Cómo se relacionan y qué no podemos concluir sobre el autor a partir de esas cosas?' },
  { name: 'guided_contextual_reading', question: 'Ahora necesito explicárselo a alguien que no sabe lingüística. Guíame con un solo ejemplo concreto del programa: qué observar, cómo interpretarlo y qué pequeño paso dar después. Usa el contexto que ya verificaste sin darme otra clase teórica larga.' },
  { name: 'check_the_guided_paraphrase', question: 'Revisa en el PDF el antecedente de «que encierra a las familias» y comprueba si tu paráfrasis sobre la seguridad conservó el sentido de la frase completa. Si hay un problema, corrige tu explicación para un principiante; si no, justifícala con el pasaje.' },
];
const contextual = process.env.DL_TEST_SCENARIO === 'contextual';
const scenarioCases = contextual ? contextualCases : dialogueCases;
const chosenNames = process.env.DL_TEST_CASE_NAMES?.split(',');
const availableCases = chosenNames ? chosenNames.map(name => {
  const found = scenarioCases.find(item => item.name === name);
  assert(found, `Unknown scenario case: ${name}`);
  return found;
}) : scenarioCases;
const caseLimit = Number(process.env.DL_TEST_CASE_LIMIT || availableCases.length);
assert(Number.isInteger(caseLimit) && caseLimit > 0 && caseLimit <= availableCases.length);
const cases = availableCases.slice(0, caseLimit);

async function main() {
  const baseURL = process.env.DL_TEST_BASE_URL;
  assert(baseURL, 'Set DL_TEST_BASE_URL.');
  const output = path.resolve(process.env.DL_TEST_OUTPUT || 'output/linguist-acceptance');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.DL_TEST_BROWSER ? { executablePath: process.env.DL_TEST_BROWSER } : {}),
    args: ['--no-sandbox', '--disable-quic', ...(process.env.DL_TEST_DNS_RESOLVER_RULES ? [`--host-resolver-rules=${process.env.DL_TEST_DNS_RESOLVER_RULES}`] : [])],
  });
  const report = { started_at: new Date().toISOString(), base_url: baseURL, cases: [], browser_errors: [], dns_resolver_rules: process.env.DL_TEST_DNS_RESOLVER_RULES || null };
  const savedReport = path.join(output, 'conversation.json');
  if (process.env.DL_TEST_RESUME_ID && fs.existsSync(savedReport)) {
    const previous = JSON.parse(fs.readFileSync(savedReport, 'utf8'));
    if (previous.conversation_id === process.env.DL_TEST_RESUME_ID) Object.assign(report, previous);
    report.resumed_base_url = baseURL;
    report.resumed_at = new Date().toISOString();
    delete report.status;
    delete report.error;
    delete report.completed_at;
  }
  const persist = () => fs.writeFileSync(path.join(output, 'conversation.json'), JSON.stringify(report, null, 2));
  try {
    const context = await browser.newContext({
      baseURL,
      viewport: { width: 1440, height: 1000 },
      ...(process.env.DL_TEST_STORAGE_STATE ? { storageState: process.env.DL_TEST_STORAGE_STATE } : {}),
    });
    const page = await context.newPage();
    const readJSON = async pathname => {
      for (let retry = 0; ; retry += 1) {
        try {
          return await page.evaluate(async target => {
            const result = await fetch(target, { credentials: 'same-origin' });
            return { status: result.status, data: await result.json() };
          }, pathname);
        } catch (error) {
          if (retry >= 4 || !/Failed to fetch|NetworkError/.test(error.message)) throw error;
          report.network_retries = (report.network_retries || 0) + 1;
          persist();
          await page.waitForTimeout(2000);
        }
      }
    };
    page.on('pageerror', error => report.browser_errors.push(error.message));
    await page.goto('/');
    if (!process.env.DL_TEST_STORAGE_STATE) {
      assert(process.env.DL_TEST_EMAIL && process.env.DL_TEST_PASSWORD, 'Provide account credentials through the environment.');
      await page.locator('#loginEmail').fill(process.env.DL_TEST_EMAIL);
      await page.locator('#loginPassword').fill(process.env.DL_TEST_PASSWORD);
      await page.locator('#loginButton').click();
    }
    await page.waitForFunction(() => document.body.classList.contains('app-authenticated'));
    await page.locator('#app[aria-busy="false"]').waitFor();
    await page.waitForFunction(() => document.getElementById('chatPreset')?.value);
    if (process.env.DL_TEST_PROJECT) {
      const selectedLabel = await page.locator('#researchProject option:checked').textContent();
      if (selectedLabel !== process.env.DL_TEST_PROJECT) {
        await page.locator('#researchProject').selectOption({ label: process.env.DL_TEST_PROJECT });
        await page.locator('#app[aria-busy="false"]').waitFor();
      }
    }
    if (process.env.DL_TEST_RESUME_ID) {
      if (await page.locator('#chatConversation').inputValue() !== process.env.DL_TEST_RESUME_ID) {
        await page.locator('#chatConversation').selectOption(process.env.DL_TEST_RESUME_ID, { force: true });
        await page.locator('#loadConversation').evaluate(button => button.click());
      }
    } else {
      await page.locator('#newConversation').click();
      await page.locator('#app[aria-busy="false"]').waitFor();
      await page.waitForFunction(() => Array.from(document.querySelectorAll('#documents .source-toggle')).some(input => input.checked && !input.disabled));
      await page.getByRole('button', { name: 'Guíame en la lectura', exact: true }).click();
      assert((await page.locator('#chatQuestion').inputValue()).includes('No sé por dónde empezar'));
      assert.equal(await page.locator('#chatConversation').inputValue(), '', 'A starter prepares a draft without sending it.');
      await page.screenshot({ path: path.join(output, 'welcome-desktop.png'), fullPage: true });
    }
    report.model = await page.locator('#chatPreset').inputValue();
    const firstCase = Number(process.env.DL_TEST_START_CASE || (process.env.DL_TEST_RESUME_ID ? report.cases.length : 0));
    for (const testCase of cases.slice(firstCase)) {
      let started = Date.now();
      console.log(`START ${testCase.name}`);
      const priorConversation = await page.locator('#chatConversation').inputValue();
      let resumed = false;
      if (process.env.DL_TEST_RESUME_ID && priorConversation) {
        const saved = (await readJSON(`/api/conversations/${priorConversation}/messages`)).data;
        const activity = (await readJSON(`/api/conversations/${priorConversation}/activity`)).data;
        resumed = (saved.items.at(-2)?.content?.text === testCase.question && saved.items.at(-1)?.role === 'assistant') || (activity.state === 'running' && activity.question === testCase.question);
        if (resumed && activity.question === testCase.question && activity.started_at) started = activity.started_at * 1000;
      }
      if (!resumed) {
        await page.locator('#chatQuestion').fill(testCase.question);
        const creation = !priorConversation ? page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/conversations') : null;
        await page.locator('#sendChat').click();
        if (creation) {
          const response = await creation;
          assert(response.ok(), `Conversation creation failed: ${await response.text()}`);
          const created = await response.json();
          await page.waitForFunction(id => document.getElementById('chatConversation').value === id, created.id);
        }
      }
      await page.waitForFunction(() => Boolean(document.getElementById('chatConversation').value));
      report.conversation_id = await page.locator('#chatConversation').inputValue();
      persist();
      let settled = false;
      const maximumPolls = Math.ceil(Number(process.env.DL_TEST_TURN_TIMEOUT_SECONDS || 900) / 5);
      for (let poll = 0; poll < maximumPolls; poll += 1) {
        await page.waitForTimeout(5000);
        const response = await readJSON(`/api/conversations/${report.conversation_id}/messages`);
        assert.equal(response.status, 200);
        const messages = response.data;
        const user = messages.items.at(-2);
        const assistant = messages.items.at(-1);
        if (user?.content?.text === testCase.question && assistant?.role === 'assistant') {
          const result = assistant.content;
          assert(result.answer_interpretation?.trim(), 'A completed turn must have an answer.');
          assert(!/^[\s{]*"(?:kind|final_answer|tool_name)"/.test(result.answer_interpretation), 'The user must see prose, not the action envelope.');
          report.cases.push({ name: testCase.name, question: testCase.question, duration_seconds: Math.round((Date.now() - started) / 1000), resumed, result });
          persist();
          console.log(`COMPLETE ${testCase.name}: ${report.cases.at(-1).duration_seconds}s, ${(result.tool_trace || []).length} tools`);
          if (testCase.name === 'corpus_count') assert(result.tool_trace?.some(item => /^corpus_/.test(item.tool_name)), 'A count must use the corpus engine.');
          if (testCase.name === 'sustained_memory') assert(/prueba del puente/i.test(result.answer_interpretation), 'The eighth exchange must retain the criterion agreed in the seventh.');
          settled = true;
          break;
        }
        const activityResponse = await readJSON(`/api/conversations/${report.conversation_id}/activity`);
        assert.equal(activityResponse.status, 200);
        const activity = activityResponse.data;
        if (activity.state === 'error') throw new Error(activity.error || 'The model turn failed.');
        if (poll % 6 === 0) console.log(`WAIT ${testCase.name}: ${activity.state}, step ${activity.step || 0}`);
      }
      assert(settled, `Timed out waiting for ${testCase.name}.`);
      await page.waitForFunction(() => !document.getElementById('sendChat').disabled);
    }
    // One batched desktop/mobile inspection, including history reopening.
    await page.reload();
    await page.locator('#app[aria-busy="false"]').waitFor();
    if (await page.locator('#chatConversation').inputValue() !== report.conversation_id) {
      await page.locator('#chatConversation').selectOption(report.conversation_id, { force: true });
      await page.locator('#loadConversation').evaluate(button => button.click());
    }
    await page.waitForFunction(count => document.querySelectorAll('.chat-message-assistant').length >= count, cases.length);
    assert.equal(await page.locator('#chatConversation').inputValue(), report.conversation_id);
    assert(await page.locator('.conversation-tree-open[aria-current="true"]').count(), 'The saved discussion must be accessible in history.');
    await page.locator('.conversation-tree-open[aria-current="true"]').click();
    await page.waitForFunction(count => document.querySelectorAll('.chat-message-assistant').length >= count, cases.length);
    assert(await page.locator('.chat-message-assistant .message-body strong').count(), 'Assistant emphasis must render as readable text.');
    await page.screenshot({ path: path.join(output, 'conversation-desktop.png'), fullPage: true });
    const sourceDetails = page.locator('details.chat-evidence').first();
    if (await sourceDetails.count()) {
      await sourceDetails.locator('summary').click();
      assert(await sourceDetails.evaluate(element => element.open));
      await sourceDetails.locator('summary').click();
    }
    const mobile = await context.newPage();
    await mobile.setViewportSize({ width: 390, height: 844 });
    mobile.on('pageerror', error => report.browser_errors.push(error.message));
    await mobile.goto('/');
    await mobile.locator('#app[aria-busy="false"]').waitFor();
    if (await mobile.locator('#chatConversation').inputValue() !== report.conversation_id) {
      await mobile.locator('#chatConversation').selectOption(report.conversation_id, { force: true });
      await mobile.locator('#loadConversation').evaluate(button => button.click());
    }
    await mobile.waitForFunction(count => document.querySelectorAll('.chat-message-assistant').length >= count, cases.length);
    report.mobile_overflow = await mobile.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(report.mobile_overflow, false, 'The mobile workspace must not overflow horizontally.');
    await mobile.screenshot({ path: path.join(output, 'conversation-mobile.png'), fullPage: true });
    assert.deepEqual(report.browser_errors, []);
    if (contextual) {
      const first = report.cases.find(item => item.name === 'ideology_without_suggested_labels');
      assert(first, 'The original question must be tested before supplying ideological categories.');
      assert(first.result.research_context?.document_readings?.some(item => item.coverage === 'complete_document' && item.total_pages > 20), 'Global interpretation must include the whole programme, not only its introduction.');
      assert(first.result.research_context?.external_sources?.length >= 2, 'Context must include actual inspected external sources.');
      const operations = report.cases.flatMap(item => item.result.tool_trace || []);
      assert(operations.some(item => item.tool_name === 'research_web_context'), 'Discovery must use the real configured provider.');
      assert(operations.some(item => item.tool_name === 'get_linguistic_method'), 'The theoretical lens must be inspected.');
      assert(await page.locator('.message-body a[href^="https://"]').count() >= 2, 'Contextual sources must be visibly clickable.');
      assert(!report.cases.some(item => /\]\(turn\d+|cite/.test(item.result.answer_interpretation)), 'Internal search handles cannot substitute real source URLs.');
      report.semantic_review = 'pending: source and transport checks do not establish expert adequacy';
    }
    report.completed_at = new Date().toISOString();
    report.status = 'passed';
    persist();
  } catch (error) {
    report.status = 'failed';
    report.error = error.message;
    persist();
    throw error;
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
