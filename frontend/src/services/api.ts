import type {
  HealthResponse, LoanRecord, ModelName, PredictionFilters, PredictionPage,
  PredictionResponse, RecordFilters, RecordPage, SavedPrediction, StoredLoanRecord,
} from '../types'

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000' : window.location.origin)
).replace(/\/$/, '')

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

const statusMessages: Record<number, string> = {
  404: 'The requested record was not found. Check its ID and try again.',
  422: 'Check the supplied values and try again.',
  503: 'The service is unavailable. Check that the models and database are ready.',
}

function detailMessage(detail: unknown): string | undefined {
  if (typeof detail === 'string') return detail
  if (!Array.isArray(detail)) return undefined
  return detail.map((item: { loc?: (string | number)[]; msg?: string }) => {
    const field = item.loc?.filter((part) => part !== 'body').join(' › ')
    return `${field ? `${field}: ` : ''}${item.msg || 'Invalid value'}`
  }).join('; ')
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const timeout = AbortSignal.timeout(30_000)
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      signal: options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
      headers: { Accept: 'application/json', ...options.headers },
    })
  } catch (error) {
    if (isAbortError(error)) throw error
    if (timeout.aborted) throw new ApiError(0, 'The request timed out. Try again.')
    throw new ApiError(0, 'Cannot reach the API. Check the server connection and try again.')
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const message = detailMessage(body?.detail) || statusMessages[response.status] || 'The request failed. Try again.'
    throw new ApiError(response.status, `HTTP ${response.status}: ${message}`)
  }
  return body as T
}

function queryString(filters: object): string {
  const params = new URLSearchParams()
  for (const [name, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== '') params.set(name, String(value))
  }
  return params.size ? `?${params}` : ''
}

function jsonBody(body: unknown): RequestInit {
  return { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

export const getHealth = (signal?: AbortSignal) => request<HealthResponse>('/health', { signal })
export const getRecords = (filters: RecordFilters = {}, signal?: AbortSignal) =>
  request<RecordPage>(`/records${queryString(filters)}`, { signal })
export const getRecord = (id: number, signal?: AbortSignal) => request<StoredLoanRecord>(`/records/${id}`, { signal })
export const predict = (records: LoanRecord[], model?: ModelName | null, signal?: AbortSignal) =>
  request<PredictionResponse>('/predict', { method: 'POST', ...jsonBody({ records, model: model || null }), signal })
export const predictRecords = (record_ids: number[], model?: ModelName | null, signal?: AbortSignal) =>
  request<PredictionResponse>('/predict/records', { method: 'POST', ...jsonBody({ record_ids, model: model || null }), signal })
export const getPredictions = (filters: PredictionFilters = {}, signal?: AbortSignal) =>
  request<PredictionPage>(`/predictions${queryString(filters)}`, { signal })
export const recordOutcome = (id: number, observed_label: 0 | 1, signal?: AbortSignal) =>
  request<SavedPrediction>(`/predictions/${id}/outcome`, { method: 'PATCH', ...jsonBody({ observed_label }), signal })

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}
export function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'An unexpected error occurred. Try again.'
}
