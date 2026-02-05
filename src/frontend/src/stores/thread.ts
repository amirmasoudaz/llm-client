// State management for Thread and Workflow

import { createContext, useContext, useReducer, type ReactNode, type Dispatch } from 'react';
import type { ThreadState, WorkflowState, WorkflowEvent, ActionRequiredEvent } from '../api/types';

interface State {
    thread: ThreadState;
    workflow: WorkflowState;
}

const initialState: State = {
    thread: {
        threadId: null,
        status: 'uninitialized',
        fundingRequestId: null,
        studentId: null,
        isInitialized: false,
    },
    workflow: {
        workflowId: null,
        status: 'idle',
        currentStage: null,
        events: [],
        streamedTokens: '',
        pendingGate: null,
        credits: {
            estimated: null,
            reserved: null,
            used: null,
            remaining: null,
        },
        error: null,
    },
};

type Action =
    | { type: 'SET_THREAD'; payload: Partial<ThreadState> }
    | { type: 'START_WORKFLOW'; payload: { workflowId: string } }
    | { type: 'ADD_EVENT'; payload: WorkflowEvent }
    | { type: 'APPEND_TOKEN'; payload: string }
    | { type: 'SET_STAGE'; payload: string }
    | { type: 'SET_GATE'; payload: ActionRequiredEvent | null }
    | { type: 'SET_CREDITS'; payload: Partial<WorkflowState['credits']> }
    | { type: 'COMPLETE_WORKFLOW'; payload?: { error?: string } }
    | { type: 'RESET_WORKFLOW' };

function reducer(state: State, action: Action): State {
    switch (action.type) {
        case 'SET_THREAD':
            return {
                ...state,
                thread: { ...state.thread, ...action.payload },
            };

        case 'START_WORKFLOW':
            return {
                ...state,
                workflow: {
                    ...initialState.workflow,
                    workflowId: action.payload.workflowId,
                    status: 'running',
                },
            };

        case 'ADD_EVENT':
            return {
                ...state,
                workflow: {
                    ...state.workflow,
                    events: [...state.workflow.events, action.payload],
                },
            };

        case 'APPEND_TOKEN':
            return {
                ...state,
                workflow: {
                    ...state.workflow,
                    streamedTokens: state.workflow.streamedTokens + action.payload,
                },
            };

        case 'SET_STAGE':
            return {
                ...state,
                workflow: {
                    ...state.workflow,
                    currentStage: action.payload,
                },
            };

        case 'SET_GATE':
            return {
                ...state,
                workflow: {
                    ...state.workflow,
                    pendingGate: action.payload,
                    status: action.payload ? 'waiting' : state.workflow.status,
                },
            };

        case 'SET_CREDITS':
            return {
                ...state,
                workflow: {
                    ...state.workflow,
                    credits: { ...state.workflow.credits, ...action.payload },
                },
            };

        case 'COMPLETE_WORKFLOW':
            return {
                ...state,
                workflow: {
                    ...state.workflow,
                    status: action.payload?.error ? 'error' : 'completed',
                    error: action.payload?.error || null,
                },
            };

        case 'RESET_WORKFLOW':
            return {
                ...state,
                workflow: initialState.workflow,
            };

        default:
            return state;
    }
}

const StateContext = createContext<State | null>(null);
const DispatchContext = createContext<Dispatch<Action> | null>(null);

export function ThreadProvider({ children }: { children: ReactNode }) {
    const [state, dispatch] = useReducer(reducer, initialState);

    return (
        <StateContext.Provider value= { state } >
        <DispatchContext.Provider value={ dispatch }>
            { children }
            </DispatchContext.Provider>
            </StateContext.Provider>
  );
}

export function useThreadState() {
    const context = useContext(StateContext);
    if (!context) {
        throw new Error('useThreadState must be used within a ThreadProvider');
    }
    return context;
}

export function useThreadDispatch() {
    const context = useContext(DispatchContext);
    if (!context) {
        throw new Error('useThreadDispatch must be used within a ThreadProvider');
    }
    return context;
}
