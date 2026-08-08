export interface FieldsetJobState {
	running: boolean;
	done: number;
	total: number;
	stage: string | null;
	error: string | null;
	finishedAt: number | null;
}

export const fieldsetJobs = new Map<string, FieldsetJobState>();

export function fieldsetJobKey(pair: number, estimator: string): string {
	return `fieldset:${pair}:${estimator}`;
}
