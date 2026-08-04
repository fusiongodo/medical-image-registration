export interface JobState {
	running: boolean;
	done: number;
	total: number;
	error: string | null;
	finishedAt: number | null;
	stage?: string | null;
}

export const jobs = new Map<string, JobState>();

export function jobKey(
	pair: number,
	depth: number,
	kind: 'candidates' | 'metrics' = 'candidates',
	lam = 'fft'
): string {
	const base = `${pair}:${depth}:${lam}`;
	return kind === 'candidates' ? base : `${base}:${kind}`;
}

export function rigidJobKey(pair: number, version = 'light_v1'): string {
	return `rigid:${version}:${pair}`;
}
