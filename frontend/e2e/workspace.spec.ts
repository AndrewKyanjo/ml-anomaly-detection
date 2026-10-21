import { expect, test, type Page } from '@playwright/test'

const inputs = {
  loan_amnt: 15000, term: '36 months', int_rate: 12.5, installment: 502,
  emp_length: '5 years', annual_inc: 65000, dti: 18, delinq_2yrs: 0,
  fico_range_low: 700, inq_last_6mths: 1, pub_rec: 0, revol_util: 42,
  total_acc: 20, mort_acc: 1, pub_rec_bankruptcies: 0,
}
const records = Array.from({ length: 28 }, (_, index) => ({ ...inputs, id: index + 1, dataset_id: 1, source_row: index + 1, loan_status: index % 2 }))
const prediction = {
  id: 501, record_id: 1, model: 'xgboost', model_version: 'a'.repeat(64),
  probability: 0.61, label: 1, threshold: 0.5, inputs,
  created_at: '2026-10-21T08:00:00Z', observed_label: null as number | null, observed_at: null as string | null,
}

async function mockApi(page: Page) {
  let saved = { ...prediction }
  await page.route('http://127.0.0.1:8000/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const headers = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Methods': 'GET,POST,PATCH,OPTIONS' }
    const respond = (json: unknown, status = 200) => route.fulfill({ status, json, headers })
    if (request.method() === 'OPTIONS') return respond({})
    if (url.pathname === '/health') return respond({ status: 'ok', models: ['random_forest', 'logistic_regression', 'xgboost'], default_model: 'xgboost' })
    if (url.pathname === '/records') {
      const filtered = records.filter((record) => !url.searchParams.has('loan_status') || record.loan_status === Number(url.searchParams.get('loan_status')))
      const offset = Number(url.searchParams.get('offset') || 0)
      const limit = Number(url.searchParams.get('limit') || 25)
      return respond({ total: filtered.length, items: filtered.slice(offset, offset + limit) })
    }
    if (url.pathname.startsWith('/records/')) {
      const found = records.find((record) => record.id === Number(url.pathname.split('/').at(-1)))
      return found ? respond(found) : respond({ detail: 'Record not found.' }, 404)
    }
    if (url.pathname === '/predict' || url.pathname === '/predict/records') {
      const body = request.postDataJSON()
      const ids = body.record_ids || [null]
      return respond({ model: body.model || 'xgboost', model_version: 'a'.repeat(64), threshold: 0.5, predictions: ids.map((_: unknown, index: number) => ({ id: 501 + index, label: 1, probability: 0.61 })) })
    }
    if (request.method() === 'PATCH') {
      saved = { ...saved, observed_label: request.postDataJSON().observed_label, observed_at: '2026-10-21T09:00:00Z' }
      return respond(saved)
    }
    if (url.pathname === '/predictions') {
      const items = url.searchParams.get('observed') === 'true' && saved.observed_label === null ? [] : [saved]
      return respond({ total: items.length, items })
    }
    return respond({ detail: 'Not found.' }, 404)
  })
}

test.beforeEach(async ({ page }) => { await mockApi(page) })

test('shows loaded models and real counts', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Your models are ready to work.' })).toBeVisible()
  await expect(page.getByRole('cell', { name: 'Logistic Regression', exact: false })).toBeVisible()
  await expect(page.getByText('28', { exact: true })).toBeVisible()
})

test('paginates, filters, inspects and hands selected records to a batch', async ({ page }) => {
  await page.goto('/app/records')
  await expect(page.getByRole('button', { name: '#1', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Next', exact: false }).click()
  await expect(page.getByRole('button', { name: '#26', exact: true })).toBeVisible()
  await page.getByRole('combobox', { name: 'Observed class', exact: true }).selectOption('1')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await expect(page.getByText('14 matching records')).toBeVisible()
  await page.getByRole('button', { name: '#2', exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Record #2' })).toBeVisible()
  await page.getByRole('button', { name: 'Close record details' }).click()
  await page.getByRole('checkbox', { name: 'Select record 2', exact: true }).check()
  await page.getByRole('link', { name: 'Predict 1 selected' }).click()
  await expect(page.getByLabel('Database record IDs')).toHaveValue('2')
})

test('submits all fifteen raw fields with the selected model', async ({ page }) => {
  await page.goto('/app/predict')
  await page.getByRole('button', { name: 'Fill sample inputs' }).click()
  await page.getByLabel('Prediction model').selectOption('logistic_regression')
  const sent = page.waitForRequest((request) => new URL(request.url()).pathname === '/predict' && request.method() === 'POST')
  await page.getByRole('button', { name: 'Run prediction', exact: false }).click()
  const body = (await sent).postDataJSON()
  expect(Object.keys(body.records[0])).toHaveLength(15)
  expect(body.model).toBe('logistic_regression')
  await expect(page.getByText('Prediction complete')).toBeVisible()
  await expect(page.getByText('Prediction #501')).toBeVisible()
})

test('validates every batch ID and preserves order', async ({ page }) => {
  await page.goto('/app/batch')
  await page.getByLabel('Database record IDs').fill('2, invalid, 1')
  await page.getByRole('button', { name: 'Run batch prediction', exact: false }).click()
  await expect(page.getByRole('alert')).toContainText('not a valid record ID')
  await page.getByLabel('Database record IDs').fill('2, 1, 2')
  const sent = page.waitForRequest((request) => request.url().includes('/predict/records') && request.method() === 'POST')
  await page.getByRole('button', { name: 'Run batch prediction', exact: false }).click()
  expect((await sent).postDataJSON().record_ids).toEqual([2, 1, 2])
  await expect(page.getByRole('heading', { name: '3 results' })).toBeVisible()
})

test('records an observed outcome and filters history', async ({ page }) => {
  await page.goto('/app/history')
  await page.getByRole('button', { name: 'Record observed class 0 for prediction 501' }).click()
  await expect(page.getByRole('status')).toContainText('Observed class 0 saved')
  await page.getByLabel('Observed outcomes only').check()
  await expect(page.getByText('#501', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Inputs', exact: true }).click()
  await expect(page.getByText('loan_amnt', { exact: true })).toBeVisible()
})

test('reports missing records without losing the workspace', async ({ page }) => {
  await page.goto('/app/records')
  await page.getByLabel('Find record by ID').fill('999')
  await page.getByRole('button', { name: 'Find record', exact: true }).click()
  await expect(page.getByRole('dialog').getByRole('alert')).toContainText('HTTP 404')
})

for (const status of [422, 503]) {
  test(`shows a useful HTTP ${status} prediction error`, async ({ page }) => {
    await page.route('http://127.0.0.1:8000/predict/records', (route) => route.fulfill({ status, json: { detail: status === 422 ? 'Invalid record IDs.' : 'Storage is unavailable.' }, headers: { 'Access-Control-Allow-Origin': '*' } }))
    await page.goto('/app/batch?ids=1')
    await page.getByRole('button', { name: 'Run batch prediction', exact: false }).click()
    await expect(page.getByRole('alert')).toContainText(`HTTP ${status}`)
    await expect(page.getByRole('button', { name: 'Run batch prediction', exact: false })).toBeEnabled()
  })
}
