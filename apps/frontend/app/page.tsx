'use client';

import { FormEvent, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Source = { id: string; source_type?: string; title?: string; source?: string; url?: string; text?: string; score?: number };

export default function Home() {
  const [question, setQuestion] = useState('Should an Indian retail company expand into Tier-2 cities in 2027?');
  const [context, setContext] = useState('');
  const [webSearch, setWebSearch] = useState(true);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function runResearch(e: FormEvent) {
    e.preventDefault();
    setLoading(true); setResult(null);
    try {
      const res = await fetch(`${API}/api/research`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({question, context, web_search: webSearch}) });
      setResult(await res.json());
    } catch {
      setResult({report:'Could not reach AgentOS API. Start the FastAPI service or set NEXT_PUBLIC_API_URL.', plan:[]});
    } finally { setLoading(false); }
  }

  const sources: Source[] = result?.sources || [];

  return <main>
    <nav><div className="brand"><span className="dot"/>AgentOS</div><div className="navnote">Research · Reason · Verify</div></nav>
    <section className="hero">
      <div className="eyebrow">AUTONOMOUS RESEARCH & DECISION INTELLIGENCE</div>
      <h1>Turn difficult questions into<br/><em>defensible decisions.</em></h1>
      <p className="lede">AgentOS decomposes complex problems, researches multiple evidence sources, challenges citations and produces a transparent decision brief.</p>
      <form onSubmit={runResearch} className="workspace">
        <label>RESEARCH QUESTION</label>
        <textarea value={question} onChange={e=>setQuestion(e.target.value)} rows={3}/>
        <label>OPTIONAL CONTEXT</label>
        <textarea value={context} onChange={e=>setContext(e.target.value)} placeholder="Constraints, company context, evidence notes…" rows={2}/>
        <label className="toggle"><input type="checkbox" checked={webSearch} onChange={e=>setWebSearch(e.target.checked)}/> Enable live web research</label>
        <button disabled={loading}>{loading ? 'Agents are reasoning…' : 'Start research  →'}</button>
      </form>
    </section>
    {result && <section className="result">
      <div className="status"><span className="live"/> {result.status || 'completed'} <span>Run {result.id?.slice(0,8)}</span></div>
      <div className="grid">
        <aside><h3>Agent plan</h3>{(result.plan||[]).map((p:string,i:number)=><div className="step" key={p}><b>0{i+1}</b><span>{p}</span></div>)}
          <div className="metrics"><span>Latency<strong>{result.metrics?.latency_seconds ?? '—'}s</strong></span><span>Sources<strong>{result.metrics?.source_count ?? 0}</strong></span><span>Citations<strong>{result.metrics?.citation_accuracy ?? 0}</strong></span></div>
        </aside>
        <article><h2>Decision brief</h2><div className="report">{result.report}</div></article>
      </div>
      <div className="evidence"><div className="evidence-head"><h2>Evidence trace</h2><span>{sources.length} retrieved sources</span></div>{sources.map((s)=>
        <div className="source" key={s.id}><div className="source-top"><b>{s.id}</b><span>{s.source_type || 'document'}</span>{s.score != null && <small>{s.score.toFixed(2)}</small>}</div><strong>{s.title || s.source || s.url || 'Evidence'}</strong>{s.url ? <a href={s.url} target="_blank" rel="noreferrer">Open source ↗</a> : null}<p>{s.text}</p></div>
      )}</div>
    </section>}
    <footer><span>AgentOS / AI Engineering Portfolio</span><span>Evidence-first · Observable · Evaluation-ready</span></footer>
  </main>;
}
