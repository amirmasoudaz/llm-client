// SSE Event Stream Handler for Intelligence Layer

import type { WorkflowEvent, BaseEvent } from './types';

export type SSEEventCallback = (event: WorkflowEvent) => void;
export type SSEErrorCallback = (error: Error) => void;
export type SSECompleteCallback = () => void;

const TERMINAL_EVENTS = new Set(['final_result', 'final_error', 'job_cancelled']);

export interface SSEConnection {
    close: () => void;
}

export function connectToWorkflowStream(
    workflowId: string,
    onEvent: SSEEventCallback,
    onError?: SSEErrorCallback,
    onComplete?: SSECompleteCallback
): SSEConnection {
    const url = `/v1/workflows/${workflowId}/events`;
    const eventSource = new EventSource(url);
    let closed = false;

    const handleEvent = (eventType: string) => (e: MessageEvent) => {
        if (closed) return;

        try {
            const data = JSON.parse(e.data);
            const event: WorkflowEvent = {
                ...data,
                type: eventType.replace(/_/g, '_'), // Normalize
            };
            onEvent(event);

            // Check for terminal events
            if (TERMINAL_EVENTS.has(event.type)) {
                close();
                onComplete?.();
            }
        } catch (err) {
            console.error('Failed to parse SSE event:', err, e.data);
        }
    };

    // Listen for all known event types
    const eventTypes = [
        'progress',
        'model_token',
        'action_required',
        'credits_preflight',
        'credits_reserved',
        'credits_settled',
        'final_result',
        'final_error',
        'job_cancelled',
        'ui_refresh_required',
    ];

    for (const type of eventTypes) {
        eventSource.addEventListener(type, handleEvent(type));
    }

    // Generic message handler for unknown event types
    eventSource.onmessage = (e) => {
        if (closed) return;
        try {
            const data = JSON.parse(e.data);
            const event: BaseEvent = {
                ...data,
                type: data.type || 'unknown',
            };
            onEvent(event);
        } catch {
            // Ignore parse errors for generic messages
        }
    };

    eventSource.onerror = (e) => {
        if (closed) return;
        console.error('SSE connection error:', e);
        onError?.(new Error('SSE connection failed'));
        close();
    };

    function close() {
        if (closed) return;
        closed = true;
        eventSource.close();
    }

    return { close };
}

// Helper to format event timestamps
export function formatEventTime(timestamp: number | string | null | undefined): string {
    if (!timestamp) return '--:--:--';

    const date = typeof timestamp === 'number'
        ? new Date(timestamp * 1000)
        : new Date(timestamp);

    if (isNaN(date.getTime())) return '--:--:--';

    return date.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

// Helper to get friendly stage names
export function getStageName(stage: string): string {
    const names: Record<string, string> = {
        query_received: 'Query Received',
        checking_auth: 'Authenticating',
        reserving_credits: 'Reserving Credits',
        loading_context: 'Loading Context',
        classifying_intent: 'Classifying Intent',
        building_plan: 'Building Plan',
        running_operator: 'Running Operator',
        awaiting_approval: 'Awaiting Approval',
        completed: 'Completed',
    };

    // Handle operator-specific stages like "running_operator:Email.ReviewDraft"
    if (stage.startsWith('running_operator:')) {
        const operatorName = stage.split(':')[1];
        return `Running: ${operatorName}`;
    }

    return names[stage] || stage.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
