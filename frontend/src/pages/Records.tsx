import { ArrowRight, ArrowUpRight, Search, SlidersHorizontal, X } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { Alert, EmptyState, PageHeader, Pagination } from '../components/ui'
import { getErrorMessage, getRecord, getRecords, isAbortError } from '../services/api'
import type { HealthResponse, RecordPage, StoredLoanRecord } from '../types'

interface Query { limit: number; offset: number; loan_status?: 0 | 1; dataset_id?: number }
const initialQuery: Query = { limit: 25, offset: 0 }
const number = (value: number | null | undefined, digits = 0) => value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: digits })

export default function Records({ health }: { health: HealthResponse | null }) {
  const [query, setQuery] = useState<Query>(initialQuery)
  const [status, setStatus] = useState('')
  const [dataset, setDataset] = useState('')
  const [limit, setLimit] = useState(25)
  const [page, setPage] = useState<RecordPage | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [lookup, setLookup] = useState('')
  const [detailId, setDetailId] = useState<number | null>(null)
  const [detail, setDetail] = useState<StoredLoanRecord | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const dialog = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    setIsLoading(true)
    setError(null)
    getRecords(query, controller.signal)
      .then(setPage)
      .catch((cause: unknown) => {
        if (!isAbortError(cause)) { setError(getErrorMessage(cause)); setPage(null) }
      })
      .finally(() => { if (!controller.signal.aborted) setIsLoading(false) })
    return () => controller.abort()
  }, [query])

  useEffect(() => {
    if (detailId === null) return
    const controller = new AbortController()
    setDetailLoading(true)
    setDetailError(null)
    setDetail(null)
    dialog.current?.showModal()
    getRecord(detailId, controller.signal)
      .then(setDetail)
      .catch((cause: unknown) => { if (!isAbortError(cause)) setDetailError(getErrorMessage(cause)) })
      .finally(() => { if (!controller.signal.aborted) setDetailLoading(false) })
    return () => controller.abort()
  }, [detailId])

  const applyFilters = (event: FormEvent) => {
    event.preventDefault()
    setSelected(new Set())
    setQuery({ limit, offset: 0, ...(status !== '' && { loan_status: Number(status) as 0 | 1 }), ...(dataset && { dataset_id: Number(dataset) }) })
  }
  const reset = () => { setStatus(''); setDataset(''); setLimit(25); setSelected(new Set()); setQuery({ ...initialQuery }) }
  const viewRecord = (event: FormEvent) => { event.preventDefault(); setDetailId(Number(lookup)) }
  const toggleSelected = (id: number) => setSelected((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next })
  const closeDetail = () => { dialog.current?.close(); setDetailId(null) }
  const items = page?.items ?? []
  const allSelected = items.length > 0 && items.every((record) => selected.has(record.id))
  const togglePageSelection = () => {
    const next = new Set(selected)
    if (allSelected) items.forEach((record) => next.delete(record.id))
    else {
      const additional = items.filter((record) => !next.has(record.id))
      if (next.size + additional.length > 1000) {
        setError('A batch supports up to 1,000 records. Clear some selected records before selecting this page.')
        return
      }
      additional.forEach((record) => next.add(record.id))
    }
    setSelected(next)
  }

  return (
    <>
      <PageHeader title="Records explorer" description="Find source records, inspect their inputs, and send a selection for prediction." actions={selected.size > 0 ? <Link to={`/app/batch?ids=${[...selected].join(',')}`} className="btn-primary">Predict {selected.size} selected<ArrowRight size={14} /></Link> : undefined} />
      {error && <Alert message={error} onDismiss={() => setError(null)} />}
      <section aria-label="Record filters" className="panel mb-6 p-5">
        <form onSubmit={applyFilters} className="grid items-end gap-4 sm:grid-cols-2 xl:grid-cols-[1fr_1fr_0.8fr_auto]">
          <label><span className="field-label">Observed class</span><select value={status} onChange={(event) => setStatus(event.target.value)} className="input"><option value="">All classes</option><option value="0">Class 0</option><option value="1">Class 1</option></select></label>
          <label><span className="field-label">Dataset ID</span><input className="input" type="number" min="1" step="1" placeholder="All datasets" value={dataset} onChange={(event) => setDataset(event.target.value)} /></label>
          <label><span className="field-label">Rows per page</span><select className="input" value={limit} onChange={(event) => setLimit(Number(event.target.value))}>{[10, 25, 50, 100].map((value) => <option key={value} value={value}>{value} records</option>)}</select></label>
          <div className="flex gap-2 pb-0.5"><button type="submit" className="btn-primary" disabled={isLoading}><SlidersHorizontal size={14} />Apply filters</button><button type="button" className="btn-secondary" onClick={reset} disabled={isLoading}>Reset</button></div>
        </form>
        <div className="mt-5 flex flex-col justify-between gap-4 border-t border-ink/10 pt-4 sm:flex-row sm:items-center"><p className="text-[11px] text-ink/45">{isLoading ? 'Loading records…' : `${(page?.total ?? 0).toLocaleString()} matching records`}{query.dataset_id ? ` · Dataset ${query.dataset_id}` : ''}{query.loan_status !== undefined ? ` · Class ${query.loan_status}` : ''}</p><form onSubmit={viewRecord} className="flex items-center gap-2"><label htmlFor="record-lookup" className="sr-only">Find record by ID</label><input id="record-lookup" type="number" min="1" step="1" required className="input !min-h-9 max-w-40 !py-1.5 !text-xs" placeholder="Find record ID" value={lookup} onChange={(event) => setLookup(event.target.value)} /><button type="submit" className="btn-secondary !min-h-9 !px-3 !py-1.5" disabled={detailLoading}><Search size={14} /><span className="sr-only">Find record</span></button></form></div>
      </section>

      <section aria-label="Stored records" className="panel overflow-hidden" aria-busy={isLoading}>
        <div className="overflow-x-auto"><table className="w-full text-left"><thead className="table-header"><tr><th className="w-10 px-4 py-3"><input type="checkbox" checked={allSelected} disabled={isLoading || items.length === 0} aria-label="Select all records on this page" onChange={togglePageSelection} /></th>{['Record', 'Loan amount', 'Term', 'Interest', 'Annual income', 'FICO', 'Class', ''].map((heading) => <th key={heading} className="whitespace-nowrap px-4 py-3 font-medium">{heading}</th>)}</tr></thead><tbody className="divide-y divide-ink/10">{!isLoading && items.map((record) => <tr key={record.id} className={selected.has(record.id) ? 'bg-accent/5' : 'hover:bg-ink/3'}><td className="px-4 py-4"><input type="checkbox" checked={selected.has(record.id)} onChange={() => toggleSelected(record.id)} aria-label={`Select record ${record.id}`} disabled={selected.size >= 1000 && !selected.has(record.id)} /></td><td className="table-cell"><button type="button" className="font-semibold text-accent underline-offset-4 hover:underline" onClick={() => setDetailId(record.id)}>#{record.id}</button><p className="mt-1 text-[10px] text-ink/40">Dataset {record.dataset_id}</p></td><td className="table-cell font-medium">{number(record.loan_amnt, 2)}</td><td className="table-cell text-ink/60">{record.term}</td><td className="table-cell">{number(record.int_rate, 2)}%</td><td className="table-cell">{number(record.annual_inc)}</td><td className="table-cell">{number(record.fico_range_low)}</td><td className="table-cell"><span className="rounded border border-ink/15 px-2 py-1 text-[10px]">Class {record.loan_status}</span></td><td className="table-cell"><button type="button" className="flex items-center gap-1 text-[11px] font-medium text-ink/60 hover:text-accent" onClick={() => setDetailId(record.id)}>Inspect<ArrowUpRight size={12} /></button></td></tr>)}</tbody></table></div>
        {isLoading && <div className="px-6 py-16 text-center text-sm text-ink/50" role="status">Loading your records…</div>}
        {!isLoading && !error && items.length === 0 && <EmptyState title="No records match these filters" description="Try another class or dataset ID, or reset your filters to browse all records." action={<button className="btn-secondary" type="button" onClick={reset}>Reset filters</button>} />}
        {!isLoading && error && <EmptyState title="Records could not be loaded" description="Check the API connection and try again." action={<button className="btn-secondary" type="button" onClick={() => setQuery({ ...query })}>Try again</button>} />}
        <Pagination offset={query.offset} limit={query.limit} total={page?.total ?? 0} onPageChange={(offset) => setQuery({ ...query, offset })} disabled={isLoading} />
      </section>
      <p className="mt-4 text-[11px] leading-5 text-ink/45">Class values preserve the source dataset’s labels. Loan amounts and income use the source currency.{health?.models.length ? ' Select records to run them through a loaded model.' : ''}</p>

      <dialog ref={dialog} aria-labelledby="record-detail-title" onClose={() => setDetailId(null)} onClick={(event) => { if (event.target === dialog.current && event.clientX < dialog.current.getBoundingClientRect().left) closeDetail() }}>
        <div className="sticky top-0 flex items-center justify-between border-b border-ink/10 bg-paper px-6 py-5"><div><p className="text-[11px] text-ink/45">Source record</p><h2 id="record-detail-title" className="mt-1 text-xl font-semibold tracking-tight">Record #{detailId}</h2></div><button type="button" className="rounded-md border border-ink/15 p-2" onClick={closeDetail} aria-label="Close record details"><X size={18} /></button></div>
        <div className="p-6">{detailLoading && <p role="status" className="py-8 text-sm text-ink/55">Loading record details…</p>}{detailError && <Alert message={detailError} />}{detail && <><dl className="divide-y divide-ink/10">{Object.entries(detail).map(([key, value]) => <div key={key} className="flex justify-between gap-6 py-3.5"><dt className="text-xs text-ink/55">{key.replaceAll('_', ' ')}</dt><dd className="max-w-[55%] break-words text-right text-xs font-medium">{value == null ? 'Missing' : typeof value === 'number' ? number(value, 3) : String(value)}</dd></div>)}</dl><Link to={`/app/batch?ids=${detail.id}`} onClick={closeDetail} className="btn-primary mt-6 w-full">Predict this record<ArrowRight size={14} /></Link></>}</div>
      </dialog>
    </>
  )
}
