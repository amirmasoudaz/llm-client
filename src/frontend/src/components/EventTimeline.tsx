// Event Timeline - Scrollable list of all workflow events

import { useState } from 'react';
import { formatEventTime } from '../api/sse';
import { useThreadState } from '../stores/thread';
import type { WorkflowEvent } from '../api/types';

function getEventTypeClass(type: string): string {
    const classes: Record<string, string> = {
        progress: 'progress',
        model_token: 'model_token',
        action_required: 'action_required',
        final_result: 'final_result',
        final_error: 'final_error',
        credits_preflight: 'progress',
        credits_reserved: 'progress',
        credits_settled: 'final_result',
    };
    return classes[type] || '';
}

function getEventIcon(type: string): string {
    const icons: Record<string, string> = {
        progress: '⏳',
        model_token: '💬',
        action_required: '⚠️',
        final_result: '✅',
        final_error: '❌',
        credits_preflight: '📊',
        credits_reserved: '🔒',
        credits_settled: '💰',
        job_cancelled: '🚫',
    };
    return icons[type] || '📌';
}

interface TimelineItemProps {
    event: WorkflowEvent;
}

function TimelineItem({ event }: TimelineItemProps) {
    const [expanded, setExpanded] = useState(false);

    // Don't show individual tokens in timeline (too noisy)
    if (event.type === 'model_token') {
        return null;
    }

    const payloadStr = JSON.stringify(event.data, null, 2);

    return (
        <div
            className={`timeline-item ${getEventTypeClass(event.type)} ${expanded ? 'expanded' : ''}`}
            onClick={() => setExpanded(!expanded)}
        >
            <span className="timeline-time">
                {formatEventTime(event.timestamp)}
            </span>
            <div className="timeline-content">
                <div className="timeline-type">
                    {getEventIcon(event.type)} {event.type}
                    {event.data.stage && (
                        <span className="text-muted" style={{ marginLeft: '8px', fontWeight: 400 }}>
                            → {String(event.data.stage)}
                        </span>
                    )}
                </div>
                {expanded && payloadStr !== '{}' && (
                    <pre className="timeline-payload">{payloadStr}</pre>
                )}
            </div>
        </div>
    );
}

export function EventTimeline() {
    const state = useThreadState();
    const { events } = state.workflow;

    // Filter out model_token for display
    const displayEvents = events.filter(e => e.type !== 'model_token');

    return (
        <div className="card" style={{ flex: 1 }}>
            <div className="card-header">
                <h3 className="card-title">
                    <span>📜</span>
                    Event Timeline
                </h3>
                <span className="badge badge-accent">{displayEvents.length} events</span>
            </div>

            <div className="timeline">
                {displayEvents.length === 0 ? (
                    <p className="text-muted" style={{ fontSize: '0.875rem', textAlign: 'center', padding: 'var(--spacing-lg)' }}>
                        No events yet. Submit a query to see events.
                    </p>
                ) : (
                    displayEvents.map((event) => (
                        <TimelineItem key={event.event_id} event={event} />
                    ))
                )}
            </div>
        </div>
    );
}
