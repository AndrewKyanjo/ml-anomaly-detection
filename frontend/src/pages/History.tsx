import { Fragment, useEffect, useRef, useState } from 'react'
import { Alert, EmptyState, PageHeader, Pagination } from '../components/ui'
import { getErrorMessage, getPredictions, isAbortError, recordOutcome } from '../services/api'
import type { HealthResponse, PredictionPage } from '../types'

function formatDate(value: string) {
  const date = new Date(/(?:Z|[+-]\d{2}:?\d{2})$/.test(value) ? value : `${value}Z`)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export default function History(_props: { health: HealthResponse | null }) {
  const [page, setPage] = useState<PredictionPage>({ total: 0, items: [] })
  const [offset, setOffset] = useState(0)
  const [observedOnly, setObservedOnly] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const [pendingId, setPendingId] = useState<number | null>(null)
  const [refresh, setRefresh] = useState(0)
  const feedbackRequest = useRef<AbortController | null>(null)
  const limit = 10

  useEffect(() => () => feedbackRequest.current?.abort(), [])

  useEffect(() => {
    const controller = new AbortController()
    setIsLoading(true)
    setError(null)
    getPredictions({ limit, offset, observed: observedOnly ? true : undefined }, controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return
        if (offset > 0 && response.total <= offset) setOffset(Math.max(0, Math.ceil(response.total / limit) * limit - limit))
        else setPage(response)
      })
      .catch((caught: unknown) => {
        if (!isAbortError(caught)) {
          setPage({ total: 0, items: [] })
          setError(getErrorMessage(caught))
        }
      })
      .finally(() => { if (!controller.signal.aborted) setIsLoading(false) })
    return () => controller.abort()
  }, [offset, observedOnly, refresh])

  async function saveOutcome(id: number, label: 0 | 1) {
    feedbackRequest.current?.abort()
    const controller = new AbortController()
    feedbackRequest.current = controller
    setPendingId(id)
    setError(null)
    setSuccess(null)
    try {
      const updated = await recordOutcome(id, label, controller.signal)
      setPage((current) => ({ ...current, items: current.items.map((item) => item.id === id ? updated : item) }))
      setSuccess(`Observed class ${label} saved for prediction #${id}.`)
      setRefresh((current) => current + 1)
    } catch (caught) {
      if (!isAbortError(caught)) setError(getErrorMessage(caught))
    } finally {
      if (!controller.signal.aborted) setPendingId(null)
    }
  }

  function toggleExpanded(id: number) {
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="space-y-7">
      <PageHeader title="History & feedback" description="Trace every prediction. Record observed outcomes when they become available." actions={<button className="btn-secondary" type="button" disabled={isLoading} onClick={() => setRefresh((current) => current + 1)}>{isLoading ? 'Refreshing…' : 'Refresh history ↻'}</button>} />
      {error && <Alert message={error} onDismiss={() => setError(null)} />}
      {success && <Alert message={success} kind="success" onDismiss={() => setSuccess(null)} />}
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="panel p-5"><p className="text-xs text-ink/60">Matching predictions</p><p className="mt-3 text-3xl font-semibold tracking-tight">{isLoading ? '—' : page.total.toLocaleString()}</p></div>
        <div className="panel p-5"><p className="text-xs text-ink/60">Current view</p><p className="mt-3 text-lg font-semibold">{observedOnly ? 'Observed outcomes' : 'All predictions'}</p></div>
        <div className="panel p-5"><p className="text-xs text-ink/60">Feedback policy</p><p className="mt-3 text-sm leading-6">Observed facts only. Predicted labels are never used as ground truth automatically.</p></div>
      </div>
      <section className="panel overflow-hidden" aria-busy={isLoading}>
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-ink/10 p-5 sm:px-6">
          <div><h2 className="font-semibold">Prediction log</h2><p className="mt-1 text-xs text-ink/55">Labels use the dataset’s original 0 / 1 encoding.</p></div>
          <label className="flex cursor-pointer items-center gap-2.5 text-sm"><input className="h-4 w-4 accent-accent" type="checkbox" checked={observedOnly} onChange={(event) => { setObservedOnly(event.target.checked); setOffset(0); setExpanded(new Set()) }} />Observed outcomes only</label>
        </div>
        {isLoading ? <div className="p-12 text-center text-sm text-ink/60" role="status">Loading prediction history…</div> : error && !page.items.length ? <div className="p-12 text-center text-sm text-ink/60">Could not load prediction history. Use Refresh history to retry.</div> : !page.items.length ? <div className="p-8"><EmptyState title="No predictions yet" description={observedOnly ? 'No observed outcomes have been recorded. Turn off the filter to see all predictions.' : 'Run a raw or batch prediction to begin building your prediction history.'} /></div> : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[850px] text-left text-sm">
              <thead className="border-b border-ink/10 bg-ink/[0.025] text-xs text-ink/55"><tr><th className="px-6 py-4 font-medium">Prediction / source</th><th className="px-4 py-4 font-medium">Model</th><th className="px-4 py-4 font-medium">Prediction</th><th className="px-4 py-4 font-medium">Observed outcome</th><th className="px-4 py-4 font-medium">Created</th><th className="px-6 py-4 font-medium"><span className="sr-only">Details</span></th></tr></thead>
              <tbody className="divide-y divide-ink/10">
                {page.items.map((prediction) => (
                  <Fragment key={prediction.id}>
                    <tr className="align-top">
                      <td className="px-6 py-5"><span className="font-mono font-semibold">#{prediction.id}</span><p className="mt-1 text-xs text-ink/55">{prediction.record_id == null ? 'Raw input' : `Record #${prediction.record_id}`}</p></td>
                      <td className="px-4 py-5"><p className="text-xs font-medium">{prediction.model.replaceAll('_', ' ')}</p><p className="mt-1 font-mono text-[10px] text-ink/50" title={prediction.model_version}>{prediction.model_version.slice(0, 10)}</p></td>
                      <td className="px-4 py-5"><span className="inline-flex rounded-full border border-accent/25 px-2.5 py-1 text-xs font-medium text-accent">Class {prediction.label}</span><p className="mt-1.5 text-xs text-ink/60">{(prediction.probability * 100).toFixed(1)}% class 1</p></td>
                      <td className="px-4 py-5"><div className="flex items-center gap-1.5">{([0, 1] as const).map((label) => <button key={label} className={`rounded-md border px-2.5 py-1.5 text-xs font-medium disabled:cursor-wait disabled:opacity-50 ${prediction.observed_label === label ? 'border-accent bg-accent text-paper' : 'border-ink/20 hover:border-accent hover:text-accent'}`} type="button" disabled={pendingId !== null} aria-label={`Record observed class ${label} for prediction ${prediction.id}`} aria-pressed={prediction.observed_label === label} onClick={() => saveOutcome(prediction.id, label)}>Class {label}</button>)}</div><p className="mt-2 text-[11px] text-ink/55">{pendingId === prediction.id ? 'Saving…' : prediction.observed_label == null ? 'Not yet observed' : 'Observed outcome saved'}</p></td>
                      <td className="max-w-40 px-4 py-5 text-xs leading-5 text-ink/60"><time dateTime={prediction.created_at}>{formatDate(prediction.created_at)}</time></td>
                      <td className="px-6 py-5 text-right"><button className="text-xs font-medium text-accent underline underline-offset-4" type="button" onClick={() => toggleExpanded(prediction.id)} aria-expanded={expanded.has(prediction.id)} aria-controls={`prediction-details-${prediction.id}`}>{expanded.has(prediction.id) ? 'Hide' : 'Inputs'}</button></td>
                    </tr>
                    {expanded.has(prediction.id) && <tr id={`prediction-details-${prediction.id}`}><td colSpan={6} className="bg-ink/[0.025] px-6 py-5"><div className="mb-4 flex flex-wrap gap-x-6 gap-y-2 text-xs text-ink/60"><span>Threshold: <span className="font-mono text-ink">{prediction.threshold}</span></span><span className="break-all">Model version: <span className="font-mono text-ink">{prediction.model_version}</span></span>{prediction.observed_at && <span>Observed: {formatDate(prediction.observed_at)}</span>}</div><dl className="grid gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-5">{Object.entries(prediction.inputs).map(([name, value]) => <div key={name}><dt className="font-mono text-[10px] text-ink/55">{name}</dt><dd className="mt-1 break-words text-xs font-medium">{value == null ? 'Unknown' : String(value)}</dd></div>)}</dl></td></tr>}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <Pagination offset={offset} limit={limit} total={page.total} onPageChange={setOffset} disabled={isLoading} />
      </section>
    </div>
  )
}
