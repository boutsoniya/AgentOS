import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'AgentOS — Research Intelligence', description: 'Evidence-first autonomous research and decision intelligence.' };
export default function RootLayout({children}:{children:React.ReactNode}) { return <html lang="en"><body>{children}</body></html>; }
