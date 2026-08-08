export interface BatchJobState {
	running: boolean;
	done: number;
	total: number;
	detail: string | null;
	error: string | null;
	finishedAt: number | null;
	stage?: string | null;
}

export const batchJobs = new Map<string, BatchJobState>();

export function batchJobKey(batchId: string): string {
	return `eval-batch:${batchId}`;
}
