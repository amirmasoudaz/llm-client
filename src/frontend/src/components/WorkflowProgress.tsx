// Workflow Progress Pipeline Visualization

import { PIPELINE_STAGES, type PipelineStage } from '../api/types';
import { getStageName } from '../api/sse';
import { useThreadState } from '../stores/thread';

interface ProgressStep {
    id: string;
    label: string;
    status: 'pending' | 'active' | 'completed' | 'error';
}

function getStageStatus(
    stageId: string,
    currentStage: string | null,
    workflowStatus: string
): 'pending' | 'active' | 'completed' | 'error' {
    if (!currentStage) return 'pending';

    const stages = [...PIPELINE_STAGES];
    const currentIndex = stages.findIndex(s =>
        currentStage === s || currentStage.startsWith(s)
    );
    const stageIndex = stages.indexOf(stageId as typeof PIPELINE_STAGES[number]);

    if (stageIndex < 0) return 'pending';

    if (workflowStatus === 'error' && stageIndex === currentIndex) {
        return 'error';
    }

    if (stageIndex < currentIndex) return 'completed';
    if (stageIndex === currentIndex) return 'active';
    return 'pending';
}

export function WorkflowProgress() {
    const state = useThreadState();
    const { workflow } = state;

    // Build steps from pipeline stages
    const steps: ProgressStep[] = PIPELINE_STAGES.map(stage => ({
        id: stage,
        label: getStageName(stage),
        status: getStageStatus(stage, workflow.currentStage, workflow.status),
    }));

    // If we have an operator-specific stage, show it
    const operatorStage = workflow.currentStage?.startsWith('running_operator:')
        ? workflow.currentStage
        : null;

    return (
        <div className="card" style={{ padding: 'var(--spacing-sm) var(--spacing-md)' }}>
            <div className="pipeline">
                {steps.map((step, index) => (
                    <div
                        key={step.id}
                        className={`pipeline-step ${step.status}`}
                    >
                        <div className="pipeline-dot">
                            {step.status === 'completed' && '✓'}
                            {step.status === 'active' && '●'}
                            {step.status === 'error' && '✕'}
                        </div>
                        <span className="pipeline-label">
                            {step.id === 'running_operator' && operatorStage
                                ? getStageName(operatorStage)
                                : step.label}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}
