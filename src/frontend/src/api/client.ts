// Intelligence Layer API Client

import type {
    ThreadInitRequest,
    ThreadInitResponse,
    SubmitQueryRequest,
    SubmitQueryResponse,
    ResolveActionRequest,
    ResolveActionResponse,
} from './types';

const API_BASE = '/v1';

class ApiError extends Error {
    constructor(
        public status: number,
        public detail: string
    ) {
        super(detail);
        this.name = 'ApiError';
    }
}

async function request<T>(
    method: string,
    path: string,
    body?: unknown
): Promise<T> {
    const url = `${API_BASE}${path}`;
    const options: RequestInit = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
    };

    if (body) {
        options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            const errorData = await response.json();
            detail = errorData.detail || detail;
        } catch {
            // ignore parse errors
        }
        throw new ApiError(response.status, detail);
    }

    return response.json() as Promise<T>;
}

export async function initThread(
    fundingRequestId: number,
    studentId?: number
): Promise<ThreadInitResponse> {
    const body: ThreadInitRequest = {
        funding_request_id: fundingRequestId,
        student_id: studentId,
    };
    return request<ThreadInitResponse>('POST', '/threads/init', body);
}

export async function submitQuery(
    threadId: string,
    message: string,
    queryId?: string
): Promise<SubmitQueryResponse> {
    const body: SubmitQueryRequest = {
        message,
        query_id: queryId,
    };
    return request<SubmitQueryResponse>('POST', `/threads/${threadId}/queries`, body);
}

export async function resolveAction(
    actionId: string,
    status: 'accepted' | 'declined',
    payload?: Record<string, unknown>
): Promise<ResolveActionResponse> {
    const body: ResolveActionRequest = {
        status,
        payload: payload || {},
    };
    return request<ResolveActionResponse>('POST', `/actions/${actionId}/resolve`, body);
}

export { ApiError };
