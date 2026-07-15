import { spawn, type ChildProcessWithoutNullStreams } from 'child_process';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'live_crop', 'crop_worker.py');

interface Pending {
	resolve: (payload: Record<string, unknown>) => void;
	reject: (err: Error) => void;
}

let worker: ChildProcessWithoutNullStreams | null = null;
let nextId = 1;
let stdoutBuffer = '';
const pending = new Map<number, Pending>();

function failAll(err: Error): void {
	for (const p of pending.values()) p.reject(err);
	pending.clear();
}

function ensureWorker(): ChildProcessWithoutNullStreams {
	if (worker && !worker.killed) return worker;

	const child = spawn(PYTHON, [SCRIPT], { cwd: REPO_ROOT });
	stdoutBuffer = '';

	child.stdout.on('data', (chunk: Buffer) => {
		stdoutBuffer += chunk.toString();
		let idx: number;
		while ((idx = stdoutBuffer.indexOf('\n')) >= 0) {
			const line = stdoutBuffer.slice(0, idx).trim();
			stdoutBuffer = stdoutBuffer.slice(idx + 1);
			if (!line) continue;

			let payload: Record<string, unknown>;
			try {
				payload = JSON.parse(line);
			} catch {
				continue;
			}

			const id = payload.id as number | null;
			if (id == null) continue;
			const p = pending.get(id);
			if (!p) continue;
			pending.delete(id);
			if (payload.ok) p.resolve(payload);
			else p.reject(new Error(String(payload.error ?? 'worker error')));
		}
	});

	child.stderr.on('data', (chunk: Buffer) => {
		console.error('[live-crop worker]', chunk.toString().trimEnd());
	});

	child.on('exit', (code) => {
		worker = null;
		failAll(new Error(`live-crop worker exited (code ${code})`));
	});

	child.on('error', (err) => {
		worker = null;
		failAll(err);
	});

	worker = child;
	return child;
}

export function cropRequest(req: Record<string, unknown>): Promise<Record<string, unknown>> {
	const child = ensureWorker();
	const id = nextId++;

	return new Promise((resolveFn, rejectFn) => {
		pending.set(id, { resolve: resolveFn, reject: rejectFn });
		const timeout = setTimeout(() => {
			if (pending.delete(id)) rejectFn(new Error('live-crop worker timeout'));
		}, 30000);
		const clear = () => clearTimeout(timeout);

		const wrapped: Pending = {
			resolve: (payload) => {
				clear();
				resolveFn(payload);
			},
			reject: (err) => {
				clear();
				rejectFn(err);
			}
		};
		pending.set(id, wrapped);

		child.stdin.write(JSON.stringify({ ...req, id }) + '\n');
	});
}
