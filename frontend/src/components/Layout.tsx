import { Activity, ArrowUpRight, Blocks, Database, FlaskConical, History, LayoutDashboard, Layers3, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { API_BASE_URL } from '../services/api'
import type { HealthResponse } from '../types'

const navigation = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/app/records', label: 'Records explorer', icon: Database },
  { path: '/app/predict', label: 'Raw prediction', icon: FlaskConical },
  { path: '/app/batch', label: 'Batch prediction', icon: Layers3 },
  { path: '/app/history', label: 'History & feedback', icon: History },
]

export default function Layout({ children, health, isLoading }: { children: ReactNode; health: HealthResponse | null; isLoading: boolean }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()
  const currentPage = navigation.find((item) => item.path === location.pathname)?.label ?? 'Workspace'
  const ready = health?.status === 'ok'

  return (
    <div className="min-h-screen bg-paper text-ink">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[232px] flex-col bg-ink text-paper lg:flex">
        <div className="flex items-center gap-3 px-7 pb-11 pt-9">
          <div className="flex size-9 items-center justify-center rounded-lg border border-paper/25"><Blocks size={21} strokeWidth={1.6} /></div>
          <div><span className="block text-lg font-semibold tracking-[-0.04em]">Creditcore</span><span className="text-[11px] text-paper/45">Loan classification workspace</span></div>
        </div>
        <p className="px-7 pb-3 text-[11px] text-paper/40">Workspace</p>
        <nav aria-label="Main navigation" className="space-y-1 px-4">
          {navigation.map(({ path, label, icon: Icon }) => (
            <NavLink key={path} to={path} end={path === '/'} className={({ isActive }) => `flex items-center gap-3 rounded-md px-3 py-3 text-[13px] ${isActive ? 'bg-paper text-ink font-semibold' : 'text-paper/60 hover:bg-paper/5 hover:text-paper'}`}>
              <Icon size={17} strokeWidth={1.6} aria-hidden="true" />{label}
            </NavLink>
          ))}
        </nav>
        <div className="mx-7 mt-auto border-t border-paper/15 pb-7 pt-5">
          <div className="mb-2 flex items-center gap-2 text-xs"><span className={`size-1.5 rounded-full ${ready ? 'bg-paper' : 'bg-paper/35'}`} />{isLoading ? 'Checking connection' : ready ? 'Connected to API' : 'API unavailable'}</div>
          <p className="text-[11px] leading-5 text-paper/40">Predictions and observed outcomes.<br />One traceable workspace.</p>
        </div>
      </aside>
      <div className="lg:ml-[232px]">
        <header className="flex min-h-20 items-center justify-between gap-4 border-b border-ink/10 px-5 sm:px-8 xl:px-11">
          <div className="flex items-center gap-3 text-xs">
            <button type="button" className="rounded p-1.5 lg:hidden" aria-label={menuOpen ? 'Close navigation' : 'Open navigation'} aria-expanded={menuOpen} aria-controls="mobile-navigation" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <PanelLeftClose size={21} /> : <PanelLeftOpen size={21} />}</button>
            <span className="hidden text-ink/45 sm:inline">Loan classification</span><span className="hidden text-ink/25 sm:inline">/</span><span className="font-medium">{currentPage}</span>
          </div>
          <div className="flex items-center gap-3 text-[11px]"><span className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 ${ready ? 'border-accent/20 bg-accent/5 text-accent' : 'border-ink/15 text-ink/55'}`}><Activity size={12} aria-hidden="true" />{isLoading ? 'Connecting' : ready ? 'System operational' : 'Connection required'}</span><a href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer" className="hidden items-center gap-1.5 text-ink/55 hover:text-accent sm:flex">API docs<ArrowUpRight size={13} /></a></div>
        </header>
        {menuOpen && <nav id="mobile-navigation" aria-label="Mobile navigation" className="grid border-b border-ink/10 bg-ink px-4 py-3 text-paper lg:hidden">{navigation.map(({ path, label, icon: Icon }) => <NavLink key={path} to={path} end={path === '/'} onClick={() => setMenuOpen(false)} className={({ isActive }) => `flex items-center gap-3 rounded px-3 py-3 text-sm ${isActive ? 'bg-paper text-ink' : 'text-paper/65'}`}><Icon size={17} />{label}</NavLink>)}</nav>}
        <main id="main-content" className="mx-auto max-w-[1600px] px-5 pb-10 pt-8 sm:px-8 sm:pt-10 xl:px-11">{children}</main>
        <footer className="mx-5 flex flex-wrap justify-between gap-3 border-t border-ink/10 py-5 text-[10px] text-ink/45 sm:mx-8 xl:mx-11"><span>Creditcore · Loan classification workspace</span><span>Model outputs support review. Verify observed outcomes independently.</span></footer>
      </div>
    </div>
  )
}
