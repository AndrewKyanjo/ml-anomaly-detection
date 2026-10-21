import { ChevronLeft, ChevronRight, CircleAlert, CircleCheck, Inbox, X } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ModelName } from '../types'
import { formatModelName } from './format'

export function PageHeader({ title, description, actions }: { title: string; description: string; actions?: ReactNode }) {
  return (
    <header className="mb-8 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
      <div>
        <h1 className="text-[clamp(1.8rem,3vw,2.35rem)] font-semibold leading-tight tracking-[-0.055em]">{title}</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-ink/60">{description}</p>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  )
}

export function Alert({ message, onDismiss, kind = 'error' }: { message: string; onDismiss?: () => void; kind?: 'error' | 'success' }) {
  const Icon = kind === 'error' ? CircleAlert : CircleCheck
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={`mb-5 flex items-start gap-3 rounded-lg border p-4 text-sm ${kind === 'error' ? 'border-ink/25 bg-ink/5' : 'border-accent/25 bg-accent/10'}`}>
      <Icon size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <p className="flex-1 leading-6">{message}</p>
      {onDismiss && <button type="button" aria-label="Dismiss notification" onClick={onDismiss} className="rounded p-1 hover:bg-ink/10"><X size={16} /></button>}
    </div>
  )
}

export function EmptyState({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <div className="flex min-h-56 flex-col items-center justify-center px-6 py-12 text-center">
      <Inbox size={30} strokeWidth={1.3} className="mb-4 text-ink/35" aria-hidden="true" />
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      <p className="mt-2 max-w-md text-sm leading-6 text-ink/60">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

export function Pagination({ offset, limit, total, onPageChange, disabled = false }: { offset: number; limit: number; total: number; onPageChange: (offset: number) => void; disabled?: boolean }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 border-t border-ink/10 px-5 py-4">
      <p className="text-xs text-ink/60">{total > 0 ? `${(offset + 1).toLocaleString()}–${Math.min(offset + limit, total).toLocaleString()} of ${total.toLocaleString()}` : '0 records'}</p>
      <div className="flex gap-2">
        <button type="button" className="btn-secondary !px-3 !py-2" onClick={() => onPageChange(Math.max(0, offset - limit))} disabled={disabled || offset === 0}><ChevronLeft size={14} aria-hidden="true" />Previous</button>
        <button type="button" className="btn-secondary !px-3 !py-2" onClick={() => onPageChange(offset + limit)} disabled={disabled || offset + limit >= total}>Next<ChevronRight size={14} aria-hidden="true" /></button>
      </div>
    </div>
  )
}

export function ModelSelect({ models, defaultModel, value, onChange, disabled = false }: { models: string[]; defaultModel?: string | null; value: ModelName | ''; onChange: (value: ModelName | '') => void; disabled?: boolean }) {
  return (
    <label className="block min-w-48">
      <span className="field-label">Prediction model</span>
      <select className="input" value={value} onChange={(event) => onChange(event.target.value as ModelName | '')} disabled={disabled}>
        <option value="">Default{defaultModel ? ` · ${formatModelName(defaultModel)}` : ''}</option>
        {models.map((model) => <option key={model} value={model}>{formatModelName(model)}</option>)}
      </select>
    </label>
  )
}
