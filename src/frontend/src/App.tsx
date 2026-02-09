// Main Application - Intelligence Layer Dashboard

import { ThreadProvider } from './stores/thread.tsx';
import { ThreadPanel } from './components/ThreadPanel';
import { CreditsPanel } from './components/CreditsPanel';
import { MetricsPanel } from './components/MetricsPanel';
import { ChatInterface } from './components/ChatInterface';
import { WorkflowProgress } from './components/WorkflowProgress';
import { EventTimeline } from './components/EventTimeline';
import { ActionGate } from './components/ActionGate';

function AppContent() {
    return (
        <div className="layout">
            {/* Header */}
            <header className="header">
                <h1 className="header-title">Intelligence Layer Dashboard</h1>
                <span className="text-muted" style={{ fontSize: '0.75rem' }}>
                    Layer 2 · Real-time Workflow Visualization
                </span>
            </header>

            {/* Sidebar */}
            <aside className="sidebar">
                <ThreadPanel />
                <CreditsPanel />
                <MetricsPanel />
            </aside>

            {/* Main Content */}
            <main className="main">
                <WorkflowProgress />
                <ChatInterface />
                <EventTimeline />
            </main>

            {/* Action Gate Modal (overlay) */}
            <ActionGate />
        </div>
    );
}

export default function App() {
    return (
        <ThreadProvider>
            <AppContent />
        </ThreadProvider>
    );
}
