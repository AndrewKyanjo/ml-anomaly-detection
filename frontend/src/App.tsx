import { useEffect, useState } from 'react'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { EmptyState } from './components/ui'
import Dashboard from './pages/Dashboard'
import Records from './pages/Records'
import RawPrediction from './pages/RawPrediction'
import BatchPrediction from './pages/BatchPrediction'
import History from './pages/History'
import { getErrorMessage, getHealth, isAbortError } from './services/api'
import type { HealthResponse } from './types'

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refresh, setRefresh] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setIsLoading(true)
    setError(null)
    getHealth(controller.signal)
      .then(setHealth)
      .catch((cause: unknown) => {
        if (!isAbortError(cause)) {
          setHealth(null)
          setError(getErrorMessage(cause))
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false)
      })
    return () => controller.abort()
  }, [refresh])

  return (
    <BrowserRouter>
      <Layout health={health} isLoading={isLoading}>
        <Routes>
          <Route path="/" element={<Dashboard health={health} isLoading={isLoading} error={error} onRefresh={() => setRefresh((value) => value + 1)} />} />
          <Route path="/app/records" element={<Records health={health} />} />
          <Route path="/app/predict" element={<RawPrediction health={health} />} />
          <Route path="/app/batch" element={<BatchPrediction health={health} />} />
          <Route path="/app/history" element={<History health={health} />} />
          <Route path="*" element={<EmptyState title="Page not found" description="Choose a page from the navigation to continue." action={<Link to="/" className="btn-primary">Return to overview</Link>} />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
