import { ArrowRight, Check, ChevronRight, CircleDot, Database, FlaskConical, Layers3, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { formatModelName } from '../components/format'
import { Alert, EmptyState, PageHeader } from '../components/ui'
import { getErrorMessage, getPredictions, getRecords, isAbortError } from '../services/api'
import type { HealthResponse } from '../types'

interface Props {
  health: HealthResponse | null
  isLoading: boolean
  error: string | null
  onRefresh: () => void
}

export default function Dashboard({ health, isLoading, error, onRefresh }: Props) {
  const [counts, setCounts] = useState<{ records: number; predictions: number; observed: number } | null>(null)
  const [countsLoading, setCountsLoading] = useState(true)
  const [countsError, setCountsError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setCountsLoading(true)
    setCountsError(null)
    Promise.all([
      getRecords({ limit: 1 }, controller.signal),
      getPredictions({ limit: 1 }, controller.signal),
      getPredictions({ limit: 1, observed: true }, controller.signal),
    ])
      .then(([records, predictions, observed]) => setCounts({ records: records.total, predictions: predictions.total, observed: observed.total }))
      .catch((cause: unknown) => {
        if (!isAbortError(cause)) {
          setCounts(null)
          setCountsError(getErrorMessage(cause))
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setCountsLoading(false)
      })
    return () => controller.abort()
  }, [revision])

  const refresh = () => {
    onRefresh()
    setRevision((value) => value + 1)
  }
  const ready = health?.status === 'ok'
  const models = health?.models ?? []
  const countValue = (value: number | undefined) => countsLoading ? '…' : value === undefined ? '—' : value.toLocaleString()

  return (
    <>
      <PageHeader title="A clear view of every decision." description="Inspect your records, run a prediction, and bring observed outcomes back into the picture." actions={<button className="btn-secondary" type="button" onClick={refresh} disabled={isLoading || countsLoading}><RefreshCw size={14} aria-hidden="true" />{isLoading || countsLoading ? 'Refreshing…' : 'Refresh status'}</button>} />
      {error && <Alert message={error} />}
      {countsError && <Alert message={`Workspace totals could not be loaded. ${countsError}`} />}

      <section aria-labelledby="system-title" className="mb-6 flex flex-col gap-6 rounded-xl bg-ink p-6 text-paper sm:p-7 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-start gap-4">
          <div className="mt-0.5 flex size-11 shrink-0 items-center justify-center rounded-full border border-paper/20"><CircleDot size={21} strokeWidth={1.5} aria-hidden="true" /></div>
          <div><h2 id="system-title" className="text-xl font-medium tracking-tight">{isLoading ? 'Connecting to your workspace' : ready ? 'Your models are ready to work.' : 'Your workspace needs a connection.'}</h2><p className="mt-2 max-w-xl text-xs leading-6 text-paper/55">{ready ? 'The prediction service and record store are available. Every new prediction is saved with its model version.' : 'Check that the API is running, then refresh the status to reconnect.'}</p></div>
        </div>
        <div className="shrink-0 border-t border-paper/15 pt-4 xl:border-l xl:border-t-0 xl:pl-8 xl:pt-0"><p className="text-[11px] text-paper/45">Default model</p><p className="mt-2 text-lg font-medium tracking-tight">{health?.default_model ? formatModelName(health.default_model) : 'Not available'}</p></div>
      </section>

      <section aria-label="Workspace totals" className="mb-9 grid grid-cols-1 divide-y divide-ink/10 rounded-xl border border-ink/12 sm:grid-cols-3 sm:divide-x sm:divide-y-0">
        {[
          { label: 'Records in your store', value: countValue(counts?.records), note: 'Imported source records', icon: Database },
          { label: 'Predictions recorded', value: countValue(counts?.predictions), note: 'A traceable inference history', icon: Layers3 },
          { label: 'Observed outcomes', value: countValue(counts?.observed), note: 'Ready for future model review', icon: Check },
        ].map(({ label, value, note, icon: Icon }) => <div key={label} className="px-6 py-6"><div className="flex items-center justify-between text-xs text-ink/60"><span>{label}</span><Icon size={16} strokeWidth={1.5} aria-hidden="true" /></div><p className="my-3 text-[2.1rem] font-semibold leading-none tracking-[-0.055em] tabular-nums" aria-live="polite">{value}</p><p className="text-[11px] text-ink/45">{note}</p></div>)}
      </section>

      <div className="grid gap-8 xl:grid-cols-[1.6fr_1fr]">
        <section aria-labelledby="models-title">
          <div className="mb-4 flex items-center justify-between"><h2 id="models-title" className="text-base font-semibold tracking-tight">Loaded models</h2><span className="text-[11px] text-ink/45">{models.length} available</span></div>
          <div className="panel overflow-hidden">
            {models.length > 0 ? <div className="overflow-x-auto"><table className="w-full text-left"><thead className="table-header"><tr><th className="px-5 py-3 font-medium">Model</th><th className="px-4 py-3 font-medium">Status</th><th className="px-5 py-3 text-right font-medium">Role</th></tr></thead><tbody className="divide-y divide-ink/10">{models.map((model) => <tr key={model}><td className="px-5 py-5"><p className="text-[13px] font-semibold">{formatModelName(model)}</p><p className="mt-1 text-[10px] text-ink/45">{model}</p></td><td className="px-4 py-5 text-[11px] text-accent"><span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-accent" />Loaded</span></td><td className="px-5 py-5 text-right"><span className={`rounded px-2 py-1 text-[10px] ${model === health?.default_model ? 'bg-accent/10 font-semibold text-accent' : 'text-ink/45'}`}>{model === health?.default_model ? 'Default' : 'Alternative'}</span></td></tr>)}</tbody></table></div> : <EmptyState title={isLoading ? 'Loading models…' : 'No models available'} description="A trained model must be loaded by the prediction service." />}
          </div>
          <p className="mt-3 text-[11px] leading-5 text-ink/45">Choose a different loaded model when making a prediction. The default is selected by the saved model bundle.</p>
        </section>

        <section aria-labelledby="next-title">
          <h2 id="next-title" className="mb-4 text-base font-semibold tracking-tight">Start an analysis</h2>
          <div className="panel overflow-hidden divide-y divide-ink/10">
            {[
              { to: '/app/predict', title: 'Predict a single record', text: 'Enter the 15 inputs and compare a model’s output.', icon: FlaskConical },
              { to: '/app/records', title: 'Explore your source records', text: 'Filter, inspect, and select records for a batch.', icon: Database },
              { to: '/app/history', title: 'Review prediction history', text: 'Record an observed class when it becomes available.', icon: Layers3 },
            ].map(({ to, title, text, icon: Icon }) => <Link key={to} to={to} className="flex items-start gap-3 px-5 py-5 hover:bg-accent/5"><Icon size={18} strokeWidth={1.5} className="mt-0.5 text-accent" aria-hidden="true" /><div className="flex-1"><h3 className="text-xs font-semibold">{title}</h3><p className="mt-1.5 max-w-xs text-[11px] leading-5 text-ink/55">{text}</p></div><ChevronRight size={14} className="mt-0.5 text-ink/35" aria-hidden="true" /></Link>)}
          </div>
          <Link to="/app/batch" className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-accent">Have record IDs? Run a batch<ArrowRight size={14} aria-hidden="true" /></Link>
        </section>
      </div>
    </>
  )
}
