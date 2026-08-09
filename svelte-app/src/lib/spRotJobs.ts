export interface SpRotJobState {
	running: boolean;
	done: number;
	total: number;
	detail: string | null;
	error: string | null;
	finishedAt: number | null;
	stage?: string | null;
	failed?: number;
	skipped?: number;
}

export const spRotJobs = new Map<string, SpRotJobState>();

export function spRotJobKey(runId: string): string {
	return `sp-rot:${runId}`;
}
