import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Alert, ModelSelect, PageHeader } from '../components/ui'
import PredictionResults from '../components/PredictionResults'
import { getErrorMessage, isAbortError, predictRecords } from '../services/api'
import type { HealthResponse, ModelName, PredictionResponse } from '../types'

function parseIds(value: string): number[] {
  const tokens = value.trim().split(/[\s,]+/).filter(Boolean)
  if (!tokens.length) throw new Error('Enter at least one database record ID.')
  if (tokens.length > 1000) throw new Error('A batch can contain at most 1,000 record IDs.')
  return tokens.map((token) => {
    const id = Number(token)
    if (!/^\d+$/.test(token) || !Number.isSafeInteger(id) || id < 1) {
      throw new Error(`“${token}” is not a valid record ID. Use positive whole numbers.`)
    }
    return id
  })
}

export default function BatchPrediction({ health }: { health: HealthResponse | null }) {
  const [searchParams] = useSearchParams()
  const [input, setInput] = useState(() => searchParams.get('ids') ?? '')
  const [model, setModel] = useState<ModelName | ''>('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [submittedIds, setSubmittedIds] = useState<number[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const request = useRef<AbortController | null>(null)
  const count = useMemo(() => input.trim() ? input.trim().split(/[\s,]+/).filter(Boolean).length : 0, [input])

  useEffect(() => () => request.current?.abort(), [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setResult(null)
    let ids: number[]
    try { ids = parseIds(input) } catch (caught) { setError(getErrorMessage(caught)); return }
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    setIsLoading(true)
    try {
      const response = await predictRecords(ids, model || null, controller.signal)
      setSubmittedIds(ids)
      setResult(response)
    } catch (caught) {
      if (!isAbortError(caught)) setError(getErrorMessage(caught))
    } finally {
      if (!controller.signal.aborted) setIsLoading(false)
    }
  }

  return (
    <div className="space-y-7">
      <PageHeader title="Batch prediction" description="Predict from stored records without entering their features again." />
      {error && <Alert message={error} onDismiss={() => setError(null)} />}
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-5">
          <form className="panel p-5 sm:p-7" onSubmit={submit} aria-busy={isLoading}>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div><p className="font-mono text-xs text-accent">POST /predict/records</p><h2 className="mt-3 text-lg font-semibold">Select your records</h2></div>
              <span className="rounded-full border border-ink/15 px-3 py-1.5 font-mono text-xs">{count.toLocaleString()} / 1,000 IDs</span>
            </div>
            <fieldset disabled={isLoading} className="mt-7 space-y-6">
              <div>
                <div className="mb-2 flex items-center justify-between gap-3"><label className="field-label mb-0" htmlFor="record_ids">Database record IDs</label><button className="text-xs font-medium text-accent underline underline-offset-4 disabled:opacity-50" type="button" onClick={() => { setInput(''); setError(null); setResult(null) }} disabled={isLoading}>Clear</button></div>
                <textarea className="input min-h-48 resize-y font-mono text-sm leading-7" id="record_ids" name="record_ids" rows={6} required placeholder={'1, 2, 3\n4, 5, 6'} value={input} onChange={(event) => setInput(event.target.value)} aria-describedby="record-ids-help" />
                <p id="record-ids-help" className="mt-2 text-xs leading-5 text-ink/60">Separate IDs with commas, spaces, or new lines. Every ID must exist in the records database.</p>
              </div>
              <ModelSelect models={health?.models ?? []} defaultModel={health?.default_model} value={model} onChange={setModel} disabled={isLoading} />
              <div className="flex flex-wrap items-center justify-between gap-4 border-t border-ink/10 pt-5"><span className="text-xs text-ink/55">Results follow your submitted order.</span><button className="btn-primary" type="submit" disabled={isLoading}>{isLoading ? 'Running batch…' : 'Run batch prediction'}</button></div>
            </fieldset>
          </form>
          <section className="grid gap-4 sm:grid-cols-3" aria-label="Batch processing details">
            {[['01', 'Choose records', 'Use the IDs from Records Explorer.'], ['02', 'Run inference', 'Your selected model processes the stored features.'], ['03', 'Review results', 'Each prediction is saved to history.']].map(([number, title, description]) => <div key={number} className="panel p-5"><span className="font-mono text-xs text-accent">{number}</span><h3 className="mt-4 text-sm font-semibold">{title}</h3><p className="mt-2 text-xs leading-5 text-ink/60">{description}</p></div>)}
          </section>
        </div>
        <aside className="xl:sticky xl:top-8"><PredictionResults result={result} recordIds={submittedIds} /></aside>
      </div>
    </div>
  )
}
