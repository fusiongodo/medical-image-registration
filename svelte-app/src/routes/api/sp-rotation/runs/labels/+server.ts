import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'sp_rot_bench_cli.py');
const LABELS = new Set(['pass', 'fail', 'unsure']);

function runJson(args: string[], stdin?: string): Promise<unknown> {
	return new Promise((resolveP, rejectP) => {
		const child = spawn(PYTHON, [SCRIPT, ...args], { cwd: REPO_ROOT });
		let stdout = '';
		let stderr = '';
		if (stdin != null) {
			child.stdin.write(stdin);
			child.stdin.end();
		}
		child.stdout.on('data', (c: Buffer) => (stdout += c.toString()));
		child.stderr.on('data', (c: Buffer) => (stderr += c.toString()));
		child.on('error', (err) => rejectP(err));
		child.on('close', (code) => {
			if (code !== 0) {
				const last = stderr.trim().split('\n').filter(Boolean).pop();
				return rejectP(new Error(last || stderr || `exited ${code}`));
			}
			try {
				const text = stdout.trim();
				const start = text.indexOf('{');
				const end = text.lastIndexOf('}');
				if (start < 0 || end < start) throw new Error('no JSON object in output');
				resolveP(JSON.parse(text.slice(start, end + 1)));
			} catch (e) {
				rejectP(new Error(`bad labels output: ${(e as Error).message}`));
			}
		});
	});
}

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.run_id !== 'string' || !body.run_id) {
		error(400, 'Expected { run_id, ... }');
	}
	const runId = body.run_id as string;

	if (body.clear === true) {
		try {
			const result = await runJson(['clear-labels', runId]);
			return json(result);
		} catch (e) {
			error(500, `clear labels failed: ${(e as Error).message}`);
		}
	}

	const raw = body.labels;
	if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
		error(400, 'Expected { run_id, labels: { "pair:angle": "pass"|"fail"|"unsure" } } or { clear: true }');
	}
	const labels: Record<string, string> = {};
	for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
		if (typeof v !== 'string' || !LABELS.has(v.toLowerCase())) {
			error(400, `bad label for ${k}`);
		}
		labels[k] = v.toLowerCase();
	}

	try {
		const result = await runJson(
			['save-labels', runId],
			JSON.stringify({ labels })
		);
		return json(result);
	} catch (e) {
		error(500, `save labels failed: ${(e as Error).message}`);
	}
};
