export interface JobState {
	running: boolean;
	done: number;
	total: number;
	error: string | null;
	finishedAt: number | null;
}

export const jobs = new Map<string, JobState>();

export function jobKey(pair: number, depth: number, kind: 'candidates' | 'metrics' = 'candidates'): string {
	return kind === 'candidates' ? `${pair}:${depth}` : `${pair}:${depth}:${kind}`;
}
