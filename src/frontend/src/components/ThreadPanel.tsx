// Thread Panel Component - Thread initialization and status

import { useState } from 'react';
import { initThread } from '../api/client';
import { useThreadState, useThreadDispatch } from '../stores/thread';

export function ThreadPanel() {
    const state = useThreadState();
    const dispatch = useThreadDispatch();
    const [fundingRequestId, setFundingRequestId] = useState('');
    const [studentId, setStudentId] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleInit = async () => {
        const frId = parseInt(fundingRequestId, 10);
        if (!frId || frId < 1) {
            setError('Please enter a valid Funding Request ID');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await initThread(
                frId,
                studentId ? parseInt(studentId, 10) : undefined
            );

            dispatch({
                type: 'SET_THREAD',
                payload: {
                    threadId: response.thread_id,
                    status: response.thread_status,
                    fundingRequestId: frId,
                    studentId: studentId ? parseInt(studentId, 10) : null,
                    isInitialized: true,
                },
            });
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to initialize thread');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">
                    <span>🧵</span>
                    Thread
                </h3>
                {state.thread.isInitialized && (
                    <span className="badge badge-success">Active</span>
                )}
            </div>

            {!state.thread.isInitialized ? (
                <div className="flex flex-col gap-md">
                    <div className="input-group">
                        <label className="input-label">Funding Request ID *</label>
                        <input
                            type="number"
                            className="input"
                            placeholder="e.g., 12345"
                            value={fundingRequestId}
                            onChange={(e) => setFundingRequestId(e.target.value)}
                            min={1}
                        />
                    </div>

                    <div className="input-group">
                        <label className="input-label">Student ID (optional)</label>
                        <input
                            type="number"
                            className="input"
                            placeholder="Auto-derived if empty"
                            value={studentId}
                            onChange={(e) => setStudentId(e.target.value)}
                            min={1}
                        />
                    </div>

                    {error && (
                        <div className="text-error" style={{ fontSize: '0.75rem' }}>
                            {error}
                        </div>
                    )}

                    <button
                        className="btn btn-primary"
                        onClick={handleInit}
                        disabled={loading}
                    >
                        {loading ? <span className="spinner" /> : null}
                        Initialize Thread
                    </button>
                </div>
            ) : (
                <div className="flex flex-col gap-sm">
                    <div className="credits-row">
                        <span className="credits-label">Thread ID</span>
                        <span className="credits-value font-mono">{state.thread.threadId}</span>
                    </div>
                    <div className="credits-row">
                        <span className="credits-label">Status</span>
                        <span className="badge badge-info">{state.thread.status}</span>
                    </div>
                    <div className="credits-row">
                        <span className="credits-label">Funding Request</span>
                        <span className="credits-value">{state.thread.fundingRequestId}</span>
                    </div>
                    {state.thread.studentId && (
                        <div className="credits-row">
                            <span className="credits-label">Student ID</span>
                            <span className="credits-value">{state.thread.studentId}</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
