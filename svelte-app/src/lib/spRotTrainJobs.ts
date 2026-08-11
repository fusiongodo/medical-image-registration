export interface SpRotTrainJobState {
	running: boolean;
	detail: string | null;
	error: string | null;
	finishedAt: number | null;
	cmd?: string;
	pid?: number;
}

export const spRotTrainJobs = new Map<string, SpRotTrainJobState>();

export function spRotTrainJobKey(runId: string): string {
	return `sp-rot-train:${runId}`;
}
