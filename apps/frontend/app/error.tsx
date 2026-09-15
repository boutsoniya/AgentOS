'use client';
export default function Error({reset}:{error:Error & {digest?:string};reset:()=>void}){return <main style={{padding:'60px'}}><h1>AgentOS hit an error.</h1><button onClick={()=>reset()}>Retry</button></main>}
