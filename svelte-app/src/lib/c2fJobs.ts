export interface JobState {
	running: boolean;
	done: number;
	total: number;
	error: string | null;
	finishedAt: number | null;
	stage?: string | null;
}

export const jobs = new Map<string, JobState>();

export function jobKey(pair: number, depth: number, kind: 'candidates' | 'metrics' = 'candidates'): string {
	return kind === 'candidates' ? `${pair}:${depth}` : `${pair}:${depth}:${kind}`;
}

export function rigidJobKey(pair: number, version = 'light_v1'): string {
	return `rigid:${version}:${pair}`;
}
