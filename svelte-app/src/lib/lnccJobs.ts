export interface JobState {
	running: boolean;
	done: number;
	total: number;
	error: string | null;
	finishedAt: number | null;
}

export const jobs = new Map<string, JobState>();

export function jobKey(pair: number, depth: number): string {
	return `${pair}:${depth}`;
}
