<script lang="ts">
	type Run = {
		id: string;
		name?: string;
		pairs?: number[];
		lr?: number;
		batch_size?: number;
		max_steps?: number;
		ckpt_every?: number;
		smoke_every?: number;
		full_every?: number;
		status?: {
			state?: string;
			step?: number;
			epoch?: number;
			detail?: string | null;
			error?: string | null;
			last_eval?: {
				kind?: string;
				pass_rate?: number | null;
				n_pass?: number;
				n_total?: number;
				step?: number;
			} | null;
		};
	};

	let runs = $state<Run[]>([]);
	let runId = $state<string | null>(null);
	let status = $state<Run['status'] | null>(null);
	let live = $state<{ running?: boolean; detail?: string | null; error?: string | null } | null>(
		null
	);
	let lossTail = $state<{ step?: number; loss_total?: number; loss_kp?: number; loss_desc?: number }[]>(
		[]
	);
	let evalTail = $state<Record<string, unknown>[]>([]);
	let err = $state<string | null>(null);
	let busy = $state(false);

	let newName = $state(`rot-train-${Date.now()}`);
	let pairs = $state('0,1,3,16');
	let lr = $state(0.0001);
	let batchSize = $state(4);
	let maxSteps = $state(100000);
	let ckptEvery = $state(5000);
	let smokeEvery = $state(5000);
	let fullEvery = $state(20000);

	const selected = $derived(runs.find((r) => r.id === runId) ?? null);

	async function loadRuns() {
		const r = await fetch('/api/sp-rot-train/runs');
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		runs = Array.isArray(j.runs) ? j.runs : [];
		if (!runId && runs.length) runId = runs[0].id;
	}

	async function loadStatus() {
		if (!runId) return;
		const r = await fetch(`/api/sp-rot-train/runs/status?run=${encodeURIComponent(runId)}`);
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		status = (j.status as Run['status']) ?? null;
		live = j.live ?? null;
		lossTail = Array.isArray(j.logs?.loss) ? j.logs.loss : [];
		evalTail = Array.isArray(j.logs?.eval) ? j.logs.eval : [];
		const cfg = j.config as Run | undefined;
		if (cfg?.id) {
			const i = runs.findIndex((x) => x.id === cfg.id);
			if (i >= 0) runs[i] = { ...runs[i], ...cfg, status: status ?? undefined };
			else runs = [cfg, ...runs];
		}
	}

	async function createRun() {
		busy = true;
		err = null;
		try {
			const r = await fetch('/api/sp-rot-train/runs', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newName,
					pairs,
					lr,
					batch_size: batchSize,
					max_steps: maxSteps,
					ckpt_every: ckptEvery,
					smoke_every: smokeEvery,
					full_every: fullEvery
				})
			});
			if (!r.ok) throw new Error(await r.text());
			const j = await r.json();
			await loadRuns();
			runId = j.run?.id ?? runId;
			await loadStatus();
		} catch (e) {
			err = e instanceof Error ? e.message : 'create failed';
		} finally {
			busy = false;
		}
	}

	async function control(cmd: 'run' | 'resume' | 'pause' | 'stop') {
		if (!runId) return;
		busy = true;
		err = null;
		try {
			if (cmd === 'run' || cmd === 'resume') {
				await fetch('/api/sp-rot-train/runs/config', {
					method: 'PATCH',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						run_id: runId,
						config: {
							pairs: pairs.split(',').map((x) => Number(x.trim())).filter((n) => !Number.isNaN(n)),
							lr,
							batch_size: batchSize,
							max_steps: maxSteps,
							ckpt_every: ckptEvery,
							smoke_every: smokeEvery,
							full_every: fullEvery
						}
					})
				});
			}
			const r = await fetch('/api/sp-rot-train/runs/control', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ run_id: runId, cmd })
			});
			if (!r.ok) throw new Error(await r.text());
			await loadStatus();
		} catch (e) {
			err = e instanceof Error ? e.message : 'control failed';
		} finally {
			busy = false;
		}
	}

	$effect(() => {
		void loadRuns().catch((e) => (err = e instanceof Error ? e.message : 'load failed'));
	});

	$effect(() => {
		void runId;
		if (!runId) return;
		let stop = false;
		const tick = async () => {
			if (stop) return;
			try {
				await loadStatus();
			} catch {
				/* ignore poll errors */
			}
			if (!stop) setTimeout(tick, live?.running ? 2000 : 5000);
		};
		void tick();
		return () => {
			stop = true;
		};
	});

	function pct(v: number | null | undefined) {
		if (v == null || Number.isNaN(v)) return '—';
		return `${(100 * v).toFixed(1)}%`;
	}
</script>

<div class="page">
	<header>
		<h1>SP rot-inv train</h1>
		<p class="sub">Self-warp fine-tune · B1 gate |rot|≤1° ∧ t_rel≤5.5% · extract 512 / NMS 8</p>
	</header>

	{#if err}<p class="err">{err}</p>{/if}

	<section class="card">
		<h2>New run</h2>
		<div class="grid">
			<label>Name <input bind:value={newName} /></label>
			<label>Pairs <input bind:value={pairs} /></label>
			<label>LR <input type="number" step="0.00001" bind:value={lr} /></label>
			<label>Batch <input type="number" bind:value={batchSize} /></label>
			<label>Max steps <input type="number" bind:value={maxSteps} /></label>
			<label>Ckpt every <input type="number" bind:value={ckptEvery} /></label>
			<label>Smoke every <input type="number" bind:value={smokeEvery} /></label>
			<label>Full B1 every <input type="number" bind:value={fullEvery} /></label>
		</div>
		<button class="btn primary" disabled={busy} onclick={() => void createRun()}>Create</button>
	</section>

	<section class="card">
		<h2>Runs</h2>
		<label>
			Select
			<select bind:value={runId}>
				<option value={null}>—</option>
				{#each runs as r}
					<option value={r.id}>{r.name || r.id}</option>
				{/each}
			</select>
		</label>
		{#if selected}
			<div class="actions">
				<button class="btn primary" disabled={busy || live?.running} onclick={() => void control('run')}
					>Start</button
				>
				<button class="btn" disabled={busy || live?.running} onclick={() => void control('resume')}
					>Resume</button
				>
				<button class="btn" disabled={busy || !live?.running} onclick={() => void control('pause')}
					>Pause</button
				>
				<button class="btn danger" disabled={busy} onclick={() => void control('stop')}>Stop</button>
			</div>
			<p class="meta">
				state: {status?.state ?? '—'} · step {status?.step ?? 0} · epoch {status?.epoch ?? 0}
				{#if status?.detail}<br />{status.detail}{/if}
				{#if live?.running}<br /><span class="live">job running… {live.detail ?? ''}</span>{/if}
				{#if status?.error || live?.error}<br /><span class="err">{status?.error || live?.error}</span>{/if}
			</p>
			{#if status?.last_eval}
				<p class="meta">
					last eval ({status.last_eval.kind ?? '?'} @ step {status.last_eval.step ?? '?'}):
					<strong>{pct(status.last_eval.pass_rate)}</strong>
					({status.last_eval.n_pass ?? '?'}/{status.last_eval.n_total ?? '?'})
				</p>
			{/if}
		{/if}
	</section>

	<section class="card">
		<h2>Loss (tail)</h2>
		{#if !lossTail.length}
			<p class="muted">No loss lines yet.</p>
		{:else}
			<table>
				<thead>
					<tr><th>step</th><th>total</th><th>kp</th><th>desc</th></tr>
				</thead>
				<tbody>
					{#each lossTail.slice().reverse().slice(0, 20) as row}
						<tr>
							<td>{row.step}</td>
							<td>{row.loss_total?.toFixed?.(4) ?? row.loss_total}</td>
							<td>{row.loss_kp?.toFixed?.(4) ?? row.loss_kp}</td>
							<td>{row.loss_desc?.toFixed?.(4) ?? row.loss_desc}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</section>

	<section class="card">
		<h2>Eval (tail)</h2>
		{#if !evalTail.length}
			<p class="muted">No eval yet.</p>
		{:else}
			<table>
				<thead>
					<tr><th>kind</th><th>step</th><th>pass</th><th>n</th></tr>
				</thead>
				<tbody>
					{#each evalTail.slice().reverse() as row}
						<tr>
							<td>{row.kind}</td>
							<td>{row.step}</td>
							<td>{pct(row.pass_rate as number)}</td>
							<td>{row.n_pass}/{row.n_total}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</section>
</div>

<style>
	.page {
		padding: 20px 24px 48px;
		max-width: 960px;
		color: #e8e8ef;
	}
	h1 {
		margin: 0 0 4px;
		font-size: 1.4rem;
	}
	.sub,
	.muted,
	.meta {
		color: #9aa0b4;
		font-size: 0.9rem;
	}
	.card {
		margin-top: 18px;
		padding: 14px 16px;
		border: 1px solid #2a2d3a;
		background: #12141c;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
		gap: 10px;
		margin: 10px 0;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 0.8rem;
		color: #b8bfd4;
	}
	input,
	select {
		background: #0c0e14;
		border: 1px solid #2a2d3a;
		color: #e8e8ef;
		padding: 6px 8px;
	}
	.actions {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		margin: 10px 0;
	}
	.btn {
		background: #1c2030;
		border: 1px solid #3a4055;
		color: #e8e8ef;
		padding: 6px 12px;
		cursor: pointer;
	}
	.btn.primary {
		background: #2a4a7a;
	}
	.btn.danger {
		background: #5a2430;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.err {
		color: #f87171;
	}
	.live {
		color: #86efac;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.85rem;
	}
	th,
	td {
		border-bottom: 1px solid #2a2d3a;
		padding: 4px 6px;
		text-align: left;
	}
</style>
