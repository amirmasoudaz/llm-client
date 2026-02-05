// Chat Interface - Message input and streaming response display

import { useState, useRef, useEffect } from 'react';
import { submitQuery } from '../api/client';
import { connectToWorkflowStream, type SSEConnection } from '../api/sse';
import { useThreadState, useThreadDispatch } from '../stores/thread';
import type { WorkflowEvent, ActionRequiredEvent } from '../api/types';

export function ChatInterface() {
    const state = useThreadState();
    const dispatch = useThreadDispatch();
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const sseRef = useRef<SSEConnection | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const { thread, workflow } = state;

    // Auto-scroll when tokens arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [workflow.streamedTokens]);

    // Cleanup SSE on unmount
    useEffect(() => {
        return () => {
            sseRef.current?.close();
        };
    }, []);

    const handleSubmit = async () => {
        if (!message.trim() || !thread.threadId) return;

        setLoading(true);
        setError(null);
        dispatch({ type: 'RESET_WORKFLOW' });

        try {
            const response = await submitQuery(thread.threadId, message.trim());
            dispatch({ type: 'START_WORKFLOW', payload: { workflowId: response.query_id } });

            // Connect to SSE stream
            sseRef.current?.close();
            sseRef.current = connectToWorkflowStream(
                response.query_id,
                (event: WorkflowEvent) => handleEvent(event),
                (err) => {
                    setError(err.message);
                    dispatch({ type: 'COMPLETE_WORKFLOW', payload: { error: err.message } });
                },
                () => {
                    setLoading(false);
                }
            );

            setMessage('');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to submit query');
            setLoading(false);
        }
    };

    const handleEvent = (event: WorkflowEvent) => {
        dispatch({ type: 'ADD_EVENT', payload: event });

        switch (event.type) {
            case 'progress':
                dispatch({ type: 'SET_STAGE', payload: event.data.stage as string });
                break;

            case 'model_token':
                dispatch({ type: 'APPEND_TOKEN', payload: event.data.token as string });
                break;

            case 'action_required':
                dispatch({ type: 'SET_GATE', payload: event as ActionRequiredEvent });
                break;

            case 'credits_preflight':
                dispatch({
                    type: 'SET_CREDITS',
                    payload: {
                        estimated: event.data.estimate as number,
                        remaining: event.data.remaining as number,
                    },
                });
                break;

            case 'credits_reserved':
                dispatch({
                    type: 'SET_CREDITS',
                    payload: { reserved: event.data.reserved_credits as number },
                });
                break;

            case 'credits_settled':
                dispatch({
                    type: 'SET_CREDITS',
                    payload: {
                        used: event.data.credits_used as number,
                        remaining: event.data.balance_after as number,
                        reserved: null,
                    },
                });
                break;

            case 'final_result':
                dispatch({ type: 'COMPLETE_WORKFLOW' });
                setLoading(false);
                break;

            case 'final_error':
                dispatch({ type: 'COMPLETE_WORKFLOW', payload: { error: event.data.error as string } });
                setLoading(false);
                break;
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="chat-container">
            <div className="chat-messages">
                {/* Show streamed response */}
                {workflow.streamedTokens && (
                    <div className="streaming-text">
                        {workflow.streamedTokens}
                        {workflow.status === 'running' && <span className="streaming-cursor" />}
                    </div>
                )}

                {/* Show error if any */}
                {workflow.error && (
                    <div
                        style={{
                            padding: 'var(--spacing-md)',
                            background: 'rgba(239, 68, 68, 0.1)',
                            borderRadius: 'var(--radius-md)',
                            border: '1px solid var(--color-error)',
                            marginTop: 'var(--spacing-md)',
                        }}
                    >
                        <p className="text-error font-medium">Error</p>
                        <p className="text-error" style={{ fontSize: '0.875rem', marginTop: '4px' }}>
                            {workflow.error}
                        </p>
                    </div>
                )}

                {/* Show completed message */}
                {workflow.status === 'completed' && !workflow.error && (
                    <div
                        style={{
                            padding: 'var(--spacing-md)',
                            background: 'rgba(34, 197, 94, 0.1)',
                            borderRadius: 'var(--radius-md)',
                            border: '1px solid var(--color-success)',
                            marginTop: 'var(--spacing-md)',
                        }}
                    >
                        <p className="text-success font-medium">✓ Completed</p>
                    </div>
                )}

                {/* Empty state */}
                {!workflow.streamedTokens && !workflow.error && workflow.status === 'idle' && (
                    <div style={{ textAlign: 'center', padding: 'var(--spacing-2xl)', color: 'var(--color-text-muted)' }}>
                        <p style={{ fontSize: '2rem', marginBottom: 'var(--spacing-md)' }}>💬</p>
                        <p>Send a message to start a conversation</p>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-container">
                <input
                    type="text"
                    className="input chat-input"
                    placeholder={thread.isInitialized ? 'Type your message...' : 'Initialize thread first'}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={!thread.isInitialized || loading || workflow.status === 'waiting'}
                />
                <button
                    className="btn btn-primary"
                    onClick={handleSubmit}
                    disabled={!thread.isInitialized || !message.trim() || loading || workflow.status === 'waiting'}
                >
                    {loading ? <span className="spinner" /> : 'Send'}
                </button>
            </div>

            {error && !workflow.error && (
                <div style={{ padding: 'var(--spacing-sm) var(--spacing-lg)', background: 'rgba(239, 68, 68, 0.1)' }}>
                    <p className="text-error" style={{ fontSize: '0.75rem' }}>{error}</p>
                </div>
            )}
        </div>
    );
}
