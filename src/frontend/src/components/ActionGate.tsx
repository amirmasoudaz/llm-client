// Action Gate Modal - For displaying and resolving action_required events

import { useState } from 'react';
import { resolveAction } from '../api/client';
import { useThreadState, useThreadDispatch } from '../stores/thread.tsx';

export function ActionGate() {
    const state = useThreadState();
    const dispatch = useThreadDispatch();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const gate = state.workflow.pendingGate;
    if (!gate) return null;

    const { data } = gate;
    const actionId = data.action_id;
    const actionType = data.action_type || data.gate_type || 'unknown';
    const proposal = data.proposal || {};
    const message = data.message || 'This action requires your approval to proceed.';

    const handleResolve = async (status: 'accepted' | 'declined') => {
        setLoading(true);
        setError(null);

        try {
            await resolveAction(actionId, status, {});
            dispatch({ type: 'SET_GATE', payload: null });
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to resolve action');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="gate-overlay">
            <div className="gate-modal">
                <div className="gate-header">
                    <div className="gate-icon">⚠️</div>
                    <div>
                        <h3 className="gate-title">Action Required</h3>
                        <p className="gate-subtitle">{actionType.replace(/_/g, ' ')}</p>
                    </div>
                </div>

                <div className="gate-content">
                    <p style={{ marginBottom: 'var(--spacing-md)', color: 'var(--color-text-secondary)' }}>
                        {message}
                    </p>

                    {Object.keys(proposal).length > 0 && (
                        <>
                            <p className="input-label" style={{ marginBottom: 'var(--spacing-sm)' }}>
                                Proposed Changes
                            </p>
                            <pre className="gate-proposal">
                                {JSON.stringify(proposal, null, 2)}
                            </pre>
                        </>
                    )}

                    {error && (
                        <div className="text-error" style={{ marginTop: 'var(--spacing-md)', fontSize: '0.875rem' }}>
                            {error}
                        </div>
                    )}
                </div>

                <div className="gate-actions">
                    <button
                        className="btn btn-secondary"
                        onClick={() => handleResolve('declined')}
                        disabled={loading}
                    >
                        Decline
                    </button>
                    <button
                        className="btn btn-success"
                        onClick={() => handleResolve('accepted')}
                        disabled={loading}
                    >
                        {loading ? <span className="spinner" /> : null}
                        Accept & Apply
                    </button>
                </div>
            </div>
        </div>
    );
}
