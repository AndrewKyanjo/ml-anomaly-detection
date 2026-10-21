export type ModelName = 'random_forest' | 'logistic_regression' | 'xgboost' | 'prior_baseline'

export interface HealthResponse {
  status: 'ok' | 'unavailable'
  models: string[]
  default_model: string | null
}

export interface LoanRecord {
  loan_amnt: number
  term: string
  int_rate: number
  installment: number
  emp_length: string | null
  annual_inc: number
  dti: number | null
  delinq_2yrs: number
  fico_range_low: number
  inq_last_6mths: number | null
  pub_rec: number
  revol_util: number | null
  total_acc: number
  mort_acc: number
  pub_rec_bankruptcies: number
}

export interface StoredLoanRecord extends LoanRecord {
  id: number
  dataset_id: number
  source_row: number
  loan_status: 0 | 1
}

export interface RecordPage { total: number; items: StoredLoanRecord[] }
export interface Prediction { id: number; probability: number; label: 0 | 1 }
export interface PredictionResponse {
  model: string
  model_version: string
  threshold: number
  predictions: Prediction[]
}

export interface SavedPrediction extends Prediction {
  record_id: number | null
  model: string
  model_version: string
  threshold: number
  inputs: Record<string, number | string | null>
  created_at: string
  observed_label: 0 | 1 | null
  observed_at: string | null
}

export interface PredictionPage { total: number; items: SavedPrediction[] }
export interface RecordFilters {
  limit?: number
  offset?: number
  loan_status?: 0 | 1
  dataset_id?: number
}
export interface PredictionFilters { limit?: number; offset?: number; observed?: boolean }
