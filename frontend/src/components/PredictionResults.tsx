import type { PredictionResponse } from '../types'

interface PredictionResultsProps {
  result: PredictionResponse | null
  recordIds?: number[]
}

export default function PredictionResults({ result, recordIds }: PredictionResultsProps) {
  if (!result) {
    return (
      <section className="panel flex min-h-72 flex-col justify-between p-6" aria-label="Prediction results">
        <div>
          <p className="text-xs font-semibold text-ink/60">Inference output</p>
          <h2 className="mt-3 text-xl font-semibold tracking-tight">Ready when you are.</h2>
          <p className="mt-3 max-w-sm text-sm leading-6 text-ink/65">
            Submit your inputs to view the predicted class, class 1 probability, and saved prediction ID.
          </p>
        </div>
        <p className="mt-10 border-t border-ink/10 pt-4 text-xs leading-5 text-ink/60">
          Predictions are model estimates. Class labels retain the training dataset’s original 0 / 1 encoding.
        </p>
      </section>
    )
  }

  return (
    <section className="panel overflow-hidden" aria-label="Prediction results" aria-live="polite">
      <div className="border-b border-ink/10 p-6">
        <div className="flex items-center justify-between gap-4">
          <p className="text-xs font-semibold text-accent">Prediction complete</p>
          <span className="rounded-full border border-accent/25 px-2.5 py-1 text-xs font-medium text-accent">Saved</span>
        </div>
        <h2 className="mt-3 text-xl font-semibold tracking-tight">
          {result.predictions.length === 1 ? 'Your result' : `${result.predictions.length.toLocaleString()} results`}
        </h2>
        <dl className="mt-5 grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 text-xs">
          <dt className="text-ink/60">Model</dt>
          <dd className="break-all">{result.model}</dd>
          <dt className="text-ink/60">Threshold</dt>
          <dd>{result.threshold.toFixed(3)}</dd>
          <dt className="text-ink/60">Version</dt>
          <dd className="break-all font-mono" title={result.model_version}>{result.model_version.slice(0, 16)}…</dd>
        </dl>
      </div>
      <div className="max-h-[32rem] overflow-y-auto divide-y divide-ink/10">
        {result.predictions.map((prediction, index) => (
          <div key={`${prediction.id}-${index}`} className="p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs text-ink/60">{recordIds ? `Record #${recordIds[index]}` : 'Predicted label'}</p>
                <p className="mt-1 text-2xl font-semibold tracking-tight">Class {prediction.label}</p>
              </div>
              <p className="font-mono text-xs text-ink/60">Prediction #{prediction.id}</p>
            </div>
            <div className="mt-5 flex items-baseline justify-between gap-3 text-sm">
              <span className="text-ink/65">Class 1 probability</span>
              <span className="font-semibold text-accent">{(prediction.probability * 100).toFixed(1)}%</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-ink/10" role="progressbar" aria-label={`Class 1 probability for prediction ${prediction.id}`} aria-valuenow={prediction.probability * 100} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full rounded-full bg-accent" style={{ width: `${prediction.probability * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      <p className="border-t border-ink/10 px-6 py-4 text-xs leading-5 text-ink/60">
        Each result is stored in prediction history. Add an observed outcome there when it becomes available.
      </p>
    </section>
  )
}
