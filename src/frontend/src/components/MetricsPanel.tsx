// Metrics Panel - Real-time workflow metrics

import { useThreadState } from '../stores/thread';

export function MetricsPanel() {
    const state = useThreadState();
    const { workflow } = state;

    // Count tokens from model_token events
    const tokenCount = workflow.events.filter(e => e.type === 'model_token').length;

    // Calculate duration from first to last event (if completed)
    let duration = '—';
    if (workflow.events.length >= 2) {
        const firstEvent = workflow.events[0];
        const lastEvent = workflow.events[workflow.events.length - 1];

        const start = typeof firstEvent.timestamp === 'number' ? firstEvent.timestamp * 1000 : new Date(firstEvent.timestamp).getTime();
        const end = typeof lastEvent.timestamp === 'number' ? lastEvent.timestamp * 1000 : new Date(lastEvent.timestamp).getTime();

        if (!isNaN(start) && !isNaN(end)) {
            const ms = end - start;
            if (ms < 1000) {
                duration = `${ms}ms`;
            } else {
                duration = `${(ms / 1000).toFixed(2)}s`;
            }
        }
    }

    // Get status badge
    const getStatusBadge = () => {
        switch (workflow.status) {
            case 'running':
                return <span className="badge badge-info">Running</span>;
            case 'waiting':
                return <span className="badge badge-warning">Waiting</span>;
            case 'completed':
                return <span className="badge badge-success">Completed</span>;
            case 'error':
                return <span className="badge badge-error">Error</span>;
            default:
                return <span className="badge">Idle</span>;
        }
    };

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">
                    <span>📊</span>
                    Metrics
                </h3>
                {getStatusBadge()}
            </div>

            <div className="credits-display">
                <div className="credits-row">
                    <span className="credits-label">Workflow ID</span>
                    <span className="credits-value font-mono" style={{ fontSize: '0.625rem' }}>
                        {workflow.workflowId ? workflow.workflowId.slice(0, 8) + '...' : '—'}
                    </span>
                </div>

                <div className="credits-row">
                    <span className="credits-label">Tokens Streamed</span>
                    <span className="credits-value">
                        {tokenCount > 0 ? tokenCount : '—'}
                    </span>
                </div>

                <div className="credits-row">
                    <span className="credits-label">Duration</span>
                    <span className="credits-value font-mono">
                        {duration}
                    </span>
                </div>

                <div className="credits-row">
                    <span className="credits-label">Events</span>
                    <span className="credits-value">
                        {workflow.events.length}
                    </span>
                </div>

                <div className="credits-row">
                    <span className="credits-label">Stage</span>
                    <span className="credits-value" style={{ fontSize: '0.75rem' }}>
                        {workflow.currentStage || '—'}
                    </span>
                </div>
            </div>
        </div>
    );
}
