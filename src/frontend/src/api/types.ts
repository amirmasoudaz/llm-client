// Intelligence Layer API Types

export interface ThreadInitRequest {
    funding_request_id: number;
    student_id?: number;
    client_context?: Record<string, unknown>;
}

export interface ThreadInitResponse {
    thread_id: string;
    thread_status: string;
    is_new: boolean;
    message: string;
    onboarding_gate: string;
    missing_requirements: string[];
}

export interface SubmitQueryRequest {
    message: string;
    attachments?: Record<string, unknown>[];
    query_id?: string;
}

export interface SubmitQueryResponse {
    query_id: string;
    sse_url: string;
}

export interface ResolveActionRequest {
    status: 'accepted' | 'declined';
    payload?: Record<string, unknown>;
}

export interface ResolveActionResponse {
    ok: boolean;
    message?: string;
}

// SSE Event Types
export type EventType =
    | 'progress'
    | 'model_token'
    | 'action_required'
    | 'credits_preflight'
    | 'credits_reserved'
    | 'credits_settled'
    | 'final_result'
    | 'final_error'
    | 'job_cancelled'
    | 'ui_refresh_required';

export interface BaseEvent {
    event_no: number;
    event_id: string;
    type: EventType;
    timestamp: number;
    workflow_id: string;
    thread_id: number | null;
    intent_id: string | null;
    plan_id: string | null;
    step_id: string | null;
    job_id: string | null;
    correlation_id: string;
    severity: string;
    actor: {
        type?: string;
        id?: string | number;
        role?: string;
        tenant_id?: number;
    };
    data: Record<string, unknown>;
}

export interface ProgressEvent extends BaseEvent {
    type: 'progress';
    data: {
        stage: string;
        detail?: Record<string, unknown>;
        operator_name?: string;
        attempt?: number;
    };
}

export interface ModelTokenEvent extends BaseEvent {
    type: 'model_token';
    data: {
        token: string;
        index?: number;
    };
}

export interface ActionRequiredEvent extends BaseEvent {
    type: 'action_required';
    data: {
        action_id: string;
        action_type: 'apply_platform_patch' | 'collect_profile_fields';
        gate_type?: string;
        proposal?: Record<string, unknown>;
        ui_hint?: Record<string, unknown>;
        message?: string;
        requires_user_input?: boolean;
    };
}

export interface CreditsPreflight extends BaseEvent {
    type: 'credits_preflight';
    data: {
        estimate: number;
        remaining: number;
        decision: 'proceed' | 'deny';
    };
}

export interface CreditsReserved extends BaseEvent {
    type: 'credits_reserved';
    data: {
        reservation_id: string;
        reserved_credits: number;
        expires_at: string;
    };
}

export interface CreditsSettled extends BaseEvent {
    type: 'credits_settled';
    data: {
        credits_used: number;
        credit_ledger_id: string;
        balance_after: number;
        status: string;
    };
}

export interface FinalResultEvent extends BaseEvent {
    type: 'final_result';
    data: {
        message?: string;
        result?: Record<string, unknown>;
        outcomes?: Record<string, unknown>[];
    };
}

export interface FinalErrorEvent extends BaseEvent {
    type: 'final_error';
    data: {
        error: string;
        code?: string;
        details?: Record<string, unknown>;
    };
}

export interface JobCancelledEvent extends BaseEvent {
    type: 'job_cancelled';
    data: {
        reason?: string;
    };
}

export type WorkflowEvent =
    | ProgressEvent
    | ModelTokenEvent
    | ActionRequiredEvent
    | CreditsPreflight
    | CreditsReserved
    | CreditsSettled
    | FinalResultEvent
    | FinalErrorEvent
    | JobCancelledEvent
    | BaseEvent;

// Thread State
export interface ThreadState {
    threadId: string | null;
    status: string;
    fundingRequestId: number | null;
    studentId: number | null;
    isInitialized: boolean;
}

// Workflow State
export interface WorkflowState {
    workflowId: string | null;
    status: 'idle' | 'running' | 'waiting' | 'completed' | 'error';
    currentStage: string | null;
    events: WorkflowEvent[];
    streamedTokens: string;
    pendingGate: ActionRequiredEvent | null;
    credits: {
        estimated: number | null;
        reserved: number | null;
        used: number | null;
        remaining: number | null;
    };
    error: string | null;
}

// Pipeline stages (progress visualization)
export const PIPELINE_STAGES = [
    'query_received',
    'checking_auth',
    'reserving_credits',
    'loading_context',
    'classifying_intent',
    'building_plan',
    'running_operator',
    'awaiting_approval',
    'completed',
] as const;

export type PipelineStage = typeof PIPELINE_STAGES[number] | string;
