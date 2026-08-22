export interface MakeFullJobState {
	running: boolean;
	done: number;
	total: number;
	stage: string | null;
	error: string | null;
	finishedAt: number | null;
}

export const makeFullJobs = new Map<string, MakeFullJobState>();

export function makeFullJobKey(dataset: string, pairId: number, layers: string[]): string {
	return `make-full:${dataset}:${pairId}:${layers.join(',')}`;
}
