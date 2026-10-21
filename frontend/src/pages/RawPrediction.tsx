import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Alert, ModelSelect, PageHeader } from '../components/ui'
import PredictionResults from '../components/PredictionResults'
import { getErrorMessage, isAbortError, predict } from '../services/api'
import type { HealthResponse, LoanRecord, ModelName, PredictionResponse } from '../types'

type FormValues = Record<keyof LoanRecord, string>
type NumericField = Exclude<keyof LoanRecord, 'term' | 'emp_length'>

const emptyForm: FormValues = {
  loan_amnt: '', term: '36 months', int_rate: '', installment: '', emp_length: '', annual_inc: '',
  dti: '', delinq_2yrs: '', fico_range_low: '', inq_last_6mths: '', pub_rec: '', revol_util: '',
  total_acc: '', mort_acc: '', pub_rec_bankruptcies: '',
}

const sampleForm: FormValues = {
  loan_amnt: '15000', term: '36 months', int_rate: '12.5', installment: '502', emp_length: '5 years',
  annual_inc: '65000', dti: '18', delinq_2yrs: '0', fico_range_low: '700', inq_last_6mths: '1',
  pub_rec: '0', revol_util: '42', total_acc: '20', mort_acc: '1', pub_rec_bankruptcies: '0',
}

const employmentOptions = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']

function makeRecord(values: FormValues): LoanRecord {
  function number(name: NumericField, min: number, integer = false, max = Infinity): number {
    const value = Number(values[name])
    if (!values[name].trim() || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) {
      throw new Error(`Enter a valid ${integer ? 'whole number' : 'number'} for ${name.replaceAll('_', ' ')}.`)
    }
    return value
  }
  function optionalNumber(name: NumericField, min: number, integer = false): number | null {
    return values[name].trim() ? number(name, min, integer) : null
  }
  const loanAmount = number('loan_amnt', 0)
  if (loanAmount <= 0) throw new Error('Loan amount must be greater than zero.')
  return {
    loan_amnt: loanAmount,
    term: values.term,
    int_rate: number('int_rate', 0, false, 100),
    installment: number('installment', 0),
    emp_length: (values.emp_length || null) as LoanRecord['emp_length'],
    annual_inc: number('annual_inc', 0),
    dti: optionalNumber('dti', -1),
    delinq_2yrs: number('delinq_2yrs', 0, true),
    fico_range_low: number('fico_range_low', 300, true, 850),
    inq_last_6mths: optionalNumber('inq_last_6mths', 0, true),
    pub_rec: number('pub_rec', 0, true),
    revol_util: optionalNumber('revol_util', 0),
    total_acc: number('total_acc', 0, true),
    mort_acc: number('mort_acc', 0, true),
    pub_rec_bankruptcies: number('pub_rec_bankruptcies', 0, true),
  }
}

interface NumberFieldProps {
  name: NumericField
  label: string
  hint?: string
  values: FormValues
  onChange: (name: keyof LoanRecord, value: string) => void
  min?: number
  max?: number
  optional?: boolean
  integer?: boolean
}

function NumberField({ name, label, hint, values, onChange, min = 0, max, optional = false, integer = false }: NumberFieldProps) {
  return (
    <div>
      <label className="field-label" htmlFor={name}>{label} {optional && <span className="font-normal text-ink/50">(optional)</span>}</label>
      <input className="input" id={name} name={name} type="number" inputMode={integer ? 'numeric' : 'decimal'} min={min} max={max} step={integer ? 1 : 'any'} required={!optional} value={values[name]} onChange={(event) => onChange(name, event.target.value)} aria-describedby={hint ? `${name}-hint` : undefined} />
      {hint && <p id={`${name}-hint`} className="mt-1.5 text-xs leading-5 text-ink/55">{hint}</p>}
    </div>
  )
}

export default function RawPrediction({ health }: { health: HealthResponse | null }) {
  const [values, setValues] = useState<FormValues>({ ...emptyForm })
  const [model, setModel] = useState<ModelName | ''>('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const request = useRef<AbortController | null>(null)

  useEffect(() => () => request.current?.abort(), [])

  function updateValue(name: keyof LoanRecord, value: string) {
    setValues((current) => ({ ...current, [name]: value }))
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setResult(null)
    let record: LoanRecord
    try { record = makeRecord(values) } catch (caught) { setError(getErrorMessage(caught)); return }
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    setIsLoading(true)
    try {
      setResult(await predict([record], model || null, controller.signal))
    } catch (caught) {
      if (!isAbortError(caught)) setError(getErrorMessage(caught))
    } finally {
      if (!controller.signal.aborted) setIsLoading(false)
    }
  }

  return (
    <div className="space-y-7">
      <PageHeader title="Raw prediction" description="One record. Fifteen inputs. A complete, saved model prediction." />
      {error && <Alert message={error} onDismiss={() => setError(null)} />}
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <form onSubmit={submit} className="space-y-5" aria-busy={isLoading}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="font-mono text-xs text-ink/55">POST /predict</p>
            <button type="button" className="btn-secondary" disabled={isLoading} onClick={() => { setValues({ ...sampleForm }); setResult(null); setError(null) }}>Fill sample inputs</button>
          </div>
          <fieldset disabled={isLoading} className="space-y-5">
            <section className="panel p-5 sm:p-6">
              <div className="mb-5"><h2 className="font-semibold">Loan specifications</h2><p className="mt-0.5 text-xs text-ink/55">Amount, duration, and repayment.</p></div>
              <div className="grid gap-5 sm:grid-cols-2">
                <NumberField name="loan_amnt" label="Loan amount" values={values} onChange={updateValue} hint="Greater than zero." />
                <div><label className="field-label" htmlFor="term">Term</label><select className="input" id="term" value={values.term} onChange={(event) => updateValue('term', event.target.value)}><option value="36 months">36 months</option><option value="60 months">60 months</option></select></div>
                <NumberField name="int_rate" label="Interest rate (%)" values={values} onChange={updateValue} max={100} />
                <NumberField name="installment" label="Monthly installment" values={values} onChange={updateValue} />
              </div>
            </section>
            <section className="panel p-5 sm:p-6">
              <div className="mb-5"><h2 className="font-semibold">Financial profile</h2><p className="mt-0.5 text-xs text-ink/55">Income, employment, and debt.</p></div>
              <div className="grid gap-5 sm:grid-cols-2">
                <NumberField name="annual_inc" label="Annual income" values={values} onChange={updateValue} />
                <div><label className="field-label" htmlFor="emp_length">Employment length <span className="font-normal text-ink/50">(optional)</span></label><select className="input" id="emp_length" value={values.emp_length} onChange={(event) => updateValue('emp_length', event.target.value)}><option value="">Unknown</option>{employmentOptions.map((value) => <option key={value} value={value}>{value}</option>)}</select></div>
                <NumberField name="dti" label="Debt-to-income ratio" values={values} onChange={updateValue} min={-1} optional />
              </div>
            </section>
            <section className="panel p-5 sm:p-6">
              <div className="mb-5"><h2 className="font-semibold">Credit history</h2><p className="mt-0.5 text-xs text-ink/55">Existing accounts and credit indicators.</p></div>
              <div className="grid gap-5 sm:grid-cols-2">
                <NumberField name="fico_range_low" label="FICO lower bound" values={values} onChange={updateValue} min={300} max={850} integer />
                <NumberField name="revol_util" label="Revolving utilization (%)" values={values} onChange={updateValue} optional />
                <NumberField name="delinq_2yrs" label="Delinquencies in two years" values={values} onChange={updateValue} integer />
                <NumberField name="inq_last_6mths" label="Inquiries in six months" values={values} onChange={updateValue} optional integer />
                <NumberField name="pub_rec" label="Public records" values={values} onChange={updateValue} integer />
                <NumberField name="total_acc" label="Total accounts" values={values} onChange={updateValue} integer />
                <NumberField name="mort_acc" label="Mortgage accounts" values={values} onChange={updateValue} integer />
                <NumberField name="pub_rec_bankruptcies" label="Public record bankruptcies" values={values} onChange={updateValue} integer />
              </div>
            </section>
            <div className="panel flex flex-col gap-5 p-5 sm:flex-row sm:items-end sm:justify-between sm:p-6">
              <div className="min-w-0 flex-1"><ModelSelect models={health?.models ?? []} defaultModel={health?.default_model} value={model} onChange={setModel} disabled={isLoading} /></div>
              <button className="btn-primary min-h-11 shrink-0" type="submit" disabled={isLoading}>{isLoading ? 'Running prediction…' : 'Run prediction'}</button>
            </div>
          </fieldset>
          <p className="text-xs leading-5 text-ink/55">Optional blank fields are sent as null. Sample inputs are illustrative and are not a lending recommendation.</p>
        </form>
        <aside className="xl:sticky xl:top-8"><PredictionResults result={result} /></aside>
      </div>
    </div>
  )
}
