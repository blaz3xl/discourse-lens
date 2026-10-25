/** Real account/model check of the small spending indicator. No HTTP/model mocks.
 * Requires DL_TEST_BASE_URL, DL_TEST_STORAGE_STATE, DL_TEST_RESUME_ID and
 * DL_TEST_SPENDING_AMOUNT (an explicitly temporary local budget, NOT a credit).
 * Refuses to replace an existing budget. Removes its temporary budget on exit;
 * model execution history and any saved rates are preserved for inspection.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.DL_PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const baseURL = process.env.DL_TEST_BASE_URL;
  const conversation = process.env.DL_TEST_RESUME_ID;
  const amount = process.env.DL_TEST_SPENDING_AMOUNT;
  assert(baseURL && conversation && amount && process.env.DL_TEST_STORAGE_STATE);
  const output = path.resolve(process.env.DL_TEST_OUTPUT || 'output/spending-acceptance');
  fs.mkdirSync(output, { recursive: true });
  const report = { started_at: new Date().toISOString(), base_url: baseURL, browser_errors: [], samples: [] };
  const persist = () => fs.writeFileSync(path.join(output, 'spending.json'), JSON.stringify(report, null, 2));
  const browser = await chromium.launch({ headless: true,
    ...(process.env.DL_TEST_BROWSER ? { executablePath: process.env.DL_TEST_BROWSER } : {}),
    args: ['--no-sandbox', '--disable-quic'],
  });
  const context = await browser.newContext({ baseURL, viewport: { width: 1440, height: 1000 }, storageState: process.env.DL_TEST_STORAGE_STATE });
  const page = await context.newPage();
  const request = async (pathname, method = 'GET') => page.evaluate(async ({ pathname, method }) => {
    const response = await fetch(pathname, { credentials: 'same-origin', method });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.status === 204 ? null : response.json();
  }, { pathname, method });
  let temporaryBudget = false;
  try {
    page.on('pageerror', error => report.browser_errors.push(error.message));
    await page.goto('/');
    await page.locator('#app[aria-busy="false"]').waitFor();
    await page.waitForFunction(() => document.getElementById('chatPreset').value);
    report.initial = await request('/api/spending');
    assert.equal(report.initial.status, 'unconfigured', 'Never replace the user’s own budget in a test.');
    await page.locator('#chatBalance').click();
    await page.locator('#spendingPanel').waitFor();
    await page.waitForFunction(() => document.getElementById('spendingModel').textContent.includes('gpt-6-luna'));
    const rate = report.initial.model_price.saved || report.initial.model_price.suggestion;
    assert(rate, 'Configure or review actual model rates first.');
    if (!report.initial.model_price.saved) {
      await page.locator('#spendingInputRate').fill(rate.input_per_million_tokens);
      await page.locator('#spendingOutputRate').fill(rate.output_per_million_tokens);
      await page.locator('#spendingRateCurrency').fill(rate.currency);
      const saved = page.waitForResponse(response => new URL(response.url()).pathname === '/api/spending/price' && response.request().method() === 'POST');
      await page.locator('#saveSpendingPrice').click();
      assert((await saved).ok());
      await page.waitForFunction(() => document.getElementById('spendingStatus').textContent.includes('Tarifas guardadas'));
    }
    await page.locator('#spendingAmount').fill(amount);
    await page.locator('#spendingCurrency').fill(rate.currency);
    const configured = page.waitForResponse(response => new URL(response.url()).pathname === '/api/spending/budget' && response.request().method() === 'POST');
    await page.locator('#saveSpendingBudget').click();
    assert((await configured).ok());
    temporaryBudget = true;
    await page.waitForFunction(() => document.getElementById('spendingStatus').textContent.includes('Saldo establecido'));
    report.before = await request('/api/spending');
    assert.equal(Number(report.before.remaining), Number(amount));
    await page.locator('#closeSettings').click();
    await page.locator('#chatConversation').selectOption(conversation, { force: true });
    await page.locator('#loadConversation').evaluate(button => button.click());
    await page.waitForFunction(() => document.querySelectorAll('.chat-message-assistant').length >= 2);
    await page.waitForFunction(() => !document.getElementById('sendChat').disabled);
    const priorMessages = await request(`/api/conversations/${conversation}/messages`);
    const priorAssistantId = priorMessages.items.at(-1)?.id;
    const question = 'Resume en dos frases la diferencia que ya discutimos entre una observación del texto y una inferencia ideológica. Apóyate en lo que ya analizamos y no realices búsquedas nuevas.';
    report.question = question;
    await page.locator('#chatQuestion').fill(question);
    await page.locator('#sendChat').click();
    const started = Date.now();
    console.log('START real model turn and live spending polling');
    let done = false;
    for (let i = 0; i < 180; i += 1) {
      await page.waitForTimeout(5000);
      const summary = await request('/api/spending');
      report.samples.push({ at_seconds: Math.round((Date.now() - started) / 1000), remaining: summary.remaining, spent: summary.spent, calls: summary.execution_count, badge: await page.locator('#chatBalanceValue').textContent() });
      const messages = await request(`/api/conversations/${conversation}/messages`);
      if (messages.items.at(-2)?.content?.text === question && messages.items.at(-1)?.role === 'assistant' && messages.items.at(-1)?.id !== priorAssistantId) {
        report.result = messages.items.at(-1).content;
        report.after = summary;
        done = true;
        break;
      }
      const activity = await request(`/api/conversations/${conversation}/activity`);
      assert.notEqual(activity.state, 'error', activity.error);
      if (i % 6 === 0) console.log(`WAIT ${activity.state}, ${summary.execution_count} confirmed calls`);
      persist();
    }
    assert(done, 'The real turn must complete.');
    assert.equal(report.after.status, 'ready');
    assert(Number(report.after.spent) > 0);
    assert(Number(report.after.remaining) < Number(report.before.remaining));
    await page.waitForFunction(() => !document.getElementById('sendChat').disabled);
    await page.waitForTimeout(4500);
    report.visible_balance = await page.locator('#chatBalanceValue').textContent();
    assert(report.visible_balance.includes('≈'));
    const replay = await request('/api/spending');
    assert.equal(replay.remaining, report.after.remaining, 'Refreshing must not debit a second time.');
    // One batched visual pass: desktop/mobile, composer and existing settings.
    await page.screenshot({ path: path.join(output, 'balance-desktop.png'), fullPage: true });
    await page.locator('#chatBalance').click();
    await page.locator('#spendingPanel').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, 'settings-desktop.png'), fullPage: true });
    await page.locator('#closeSettings').click();
    const mobile = await context.newPage();
    mobile.on('pageerror', error => report.browser_errors.push(error.message));
    await mobile.setViewportSize({ width: 390, height: 844 });
    await mobile.goto('/');
    await mobile.locator('#app[aria-busy="false"]').waitFor();
    await mobile.waitForFunction(() => document.getElementById('chatBalanceValue').textContent.includes('≈'));
    const geometry = await mobile.evaluate(() => {
      const balance = document.getElementById('chatBalance').getBoundingClientRect();
      const settings = document.getElementById('chatModelSettings').getBoundingClientRect();
      const send = document.getElementById('sendChat').getBoundingClientRect();
      return { overflow: document.documentElement.scrollWidth > innerWidth,
        balance: balance.toJSON(), settings: settings.toJSON(), send: send.toJSON() };
    });
    report.mobile_geometry = geometry;
    assert.equal(geometry.overflow, false);
    assert(geometry.balance.x >= geometry.settings.right && geometry.balance.right < geometry.send.x);
    await mobile.screenshot({ path: path.join(output, 'balance-mobile.png'), fullPage: true });
    await mobile.locator('#chatBalance').click();
    await mobile.locator('#spendingPanel').scrollIntoViewIfNeeded();
    report.mobile_settings_overflow = await mobile.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(report.mobile_settings_overflow, false);
    await mobile.screenshot({ path: path.join(output, 'settings-mobile.png'), fullPage: true });
    await mobile.close();
    assert.deepEqual(report.browser_errors, []);
    report.status = 'passed';
    console.log(`PASS ${report.before.remaining} -> ${report.after.remaining} ${report.before.budget.currency}; ${report.after.execution_count} model calls`);
  } catch (error) {
    report.status = 'failed';
    report.error = error.stack;
    throw error;
  } finally {
    if (temporaryBudget) {
      await request('/api/spending/budget', 'DELETE');
      report.cleaned_up = (await request('/api/spending')).status === 'unconfigured';
    }
    report.completed_at = new Date().toISOString();
    persist();
    await browser.close();
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
