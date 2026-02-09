// Credits Panel - Display credit reservation and usage

import { useThreadState } from '../stores/thread.tsx';

export function CreditsPanel() {
    const state = useThreadState();
    const { credits } = state.workflow;

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">
                    <span>💳</span>
                    Credits
                </h3>
            </div>

            <div className="credits-display">
                <div className="credits-row">
                    <span className="credits-label">Estimated</span>
                    <span className={`credits-value ${credits.estimated ? 'pending' : ''}`}>
                        {credits.estimated !== null ? credits.estimated : '—'}
                    </span>
                </div>

                <div className="credits-row">
                    <span className="credits-label">Reserved</span>
                    <span className={`credits-value ${credits.reserved ? 'pending' : ''}`}>
                        {credits.reserved !== null ? credits.reserved : '—'}
                    </span>
                </div>

                <div className="credits-row">
                    <span className="credits-label">Used</span>
                    <span className={`credits-value ${credits.used ? 'negative' : ''}`}>
                        {credits.used !== null ? credits.used : '—'}
                    </span>
                </div>

                <div className="credits-row">
                    <span className="credits-label">Remaining</span>
                    <span className={`credits-value ${credits.remaining !== null ? 'positive' : ''}`}>
                        {credits.remaining !== null ? credits.remaining : '—'}
                    </span>
                </div>
            </div>
        </div>
    );
}
