<script lang="ts">
	type Run = {
		id: string;
		name?: string;
		pairs?: number[];
		lr?: number;
		batch_size?: number;
		max_epochs?: number;
		ckpt_every_epochs?: number;
		eval_every_epochs?: number;
		log_every?: number;
		eval_max_tiles?: number;
		n_train?: number;
		n_val?: number;
		n_test?: number;
		n_total?: number;
		status?: {
			state?: string;
			step?: number;
			epoch?: number;
			detail?: string | null;
			error?: string | null;
			last_epoch_s?: number | null;
			last_eval?: {
				kind?: string;
				pass_rate?: number | null;
				n_pass?: number;
				n_total?: number;
				n_tiles?: number;
				step?: number;
				epoch?: number;
			} | null;
		};
	};

	let runs = $state<Run[]>([]);
	let runId = $state<string | null>(null);
	let status = $state<Run['status'] | null>(null);
	let live = $state<{ running?: boolean; detail?: string | null; error?: string | null } | null>(
		null
	);
	let lossTail = $state<
		{ step?: number; epoch?: number; loss_total?: number; loss_kp?: number; loss_desc?: number }[]
	>([]);
	let evalTail = $state<Record<string, unknown>[]>([]);
	let epochTail = $state<
		{
			epoch?: number;
			epoch_s?: number;
			train_loss?: number | null;
			val_loss?: number | null;
			val_loss_kp?: number | null;
			val_loss_desc?: number | null;
			step?: number;
		}[]
	>([]);
	let err = $state<string | null>(null);
	let busy = $state(false);

	let newName = $state(`rot-train-${Date.now()}`);
	let pairs = $state('0,1,3,16');
	let lr = $state(0.0001);
	let batchSize = $state(8);
	let maxEpochs = $state(50);
	let ckptEveryEpochs = $state(1);
	let evalEveryEpochs = $state(1);
	let logEvery = $state(50);
	let evalMaxTiles = $state(12);

	const selected = $derived(runs.find((r) => r.id === runId) ?? null);

	function editableConfig() {
		return {
			pairs: pairs
				.split(',')
				.map((x) => Number(x.trim()))
				.filter((n) => !Number.isNaN(n)),
			lr,
			batch_size: batchSize,
			max_epochs: maxEpochs,
			ckpt_every_epochs: ckptEveryEpochs,
			eval_every_epochs: evalEveryEpochs,
			log_every: logEvery,
			eval_max_tiles: evalMaxTiles
		};
	}

	function syncFormFromRun(cfg: Run) {
		if (cfg.pairs?.length) pairs = cfg.pairs.join(',');
		if (cfg.lr != null) lr = cfg.lr;
		if (cfg.batch_size != null) batchSize = cfg.batch_size;
		if (cfg.max_epochs != null) maxEpochs = cfg.max_epochs;
		if (cfg.ckpt_every_epochs != null) ckptEveryEpochs = cfg.ckpt_every_epochs;
		if (cfg.eval_every_epochs != null) evalEveryEpochs = cfg.eval_every_epochs;
		if (cfg.log_every != null) logEvery = cfg.log_every;
		if (cfg.eval_max_tiles != null) evalMaxTiles = cfg.eval_max_tiles;
	}

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
		epochTail = Array.isArray(j.logs?.epoch) ? j.logs.epoch : [];
		const cfg = j.config as Run | undefined;
		if (cfg?.id) {
			const i = runs.findIndex((x) => x.id === cfg.id);
			if (i >= 0) runs[i] = { ...runs[i], ...cfg, status: status ?? undefined };
			else runs = [cfg, ...runs];
			if (!live?.running) syncFormFromRun(cfg);
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
					max_epochs: maxEpochs,
					ckpt_every_epochs: ckptEveryEpochs,
					eval_every_epochs: evalEveryEpochs,
					log_every: logEvery,
					eval_max_tiles: evalMaxTiles
				})
			});
			const bodyText = await r.text();
			let j: { run?: { id?: string }; message?: string; error?: string } = {};
			try {
				j = JSON.parse(bodyText);
			} catch {
				throw new Error(bodyText || `create failed (${r.status})`);
			}
			if (!r.ok) throw new Error(j.message || j.error || bodyText || `create failed (${r.status})`);
			await loadRuns();
			runId = j.run?.id ?? runId;
			newName = `rot-train-${Date.now()}`;
			await loadStatus();
		} catch (e) {
			err = e instanceof Error ? e.message : 'create failed';
		} finally {
			busy = false;
		}
	}

	async function saveConfig() {
		if (!runId) return;
		busy = true;
		err = null;
		try {
			const r = await fetch('/api/sp-rot-train/runs/config', {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ run_id: runId, config: editableConfig() })
			});
			if (!r.ok) throw new Error(await r.text());
			await loadStatus();
		} catch (e) {
			err = e instanceof Error ? e.message : 'save config failed';
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
					body: JSON.stringify({ run_id: runId, config: editableConfig() })
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

	function fmtLoss(v: number | null | undefined, digits = 4) {
		if (v == null || Number.isNaN(v)) return '—';
		const a = Math.abs(v);
		if (a > 0 && a < 1e-3) return v.toExponential(2);
		return v.toFixed(digits);
	}

	function pct(v: number | null | undefined) {
		if (v == null || Number.isNaN(v)) return '—';
		return `${(100 * v).toFixed(1)}%`;
	}

	function fmtS(v: number | null | undefined) {
		if (v == null || Number.isNaN(v)) return '—';
		return `${v.toFixed(1)}s`;
	}
</script>

<div class="page">
	<header>
		<h1>SP rot-inv train</h1>
		<p class="sub">
			Self-warp · 80/10/10 tiles · eval on val tiles (|rot|≤1° ∧ t_rel≤5.5%) · 512 / NMS 8
		</p>
	</header>

	{#if err}<p class="err">{err}</p>{/if}

	<section class="card">
		<h2>Config</h2>
		<div class="grid">
			<label>Name <input bind:value={newName} /></label>
			<label>Pairs <input bind:value={pairs} /></label>
			<label>LR <input type="number" step="0.00001" bind:value={lr} /></label>
			<label>Batch <input type="number" bind:value={batchSize} /></label>
			<label>Max epochs <input type="number" bind:value={maxEpochs} /></label>
			<label>Ckpt every epochs <input type="number" bind:value={ckptEveryEpochs} /></label>
			<label>Eval every epochs <input type="number" bind:value={evalEveryEpochs} /></label>
			<label>Log every steps <input type="number" bind:value={logEvery} /></label>
			<label>Eval max val tiles <input type="number" bind:value={evalMaxTiles} /></label>
		</div>
		<div class="actions">
			<button class="btn primary" disabled={busy} onclick={() => void createRun()}>Create</button>
			<button class="btn" disabled={busy || !runId} onclick={() => void saveConfig()}
				>Save config</button
			>
		</div>
		<p class="muted">Pause → edit (e.g. log every) → Save / Resume. Config reloads each batch.</p>
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
				split: {selected.n_train ?? '—'}/{selected.n_val ?? '—'}/{selected.n_test ?? '—'}
				(train/val/test of {selected.n_total ?? '—'})
				<br />
				state: {status?.state ?? '—'} · completed epochs {status?.epoch ?? 0} · step {status?.step ?? 0}
				· last epoch {fmtS(status?.last_epoch_s)}
				{#if live?.running}
					<br /><span class="live">job running… {status?.detail || live.detail || ''}</span>
				{:else if status?.detail}
					<br />{status.detail}
				{/if}
				{#if status?.error || live?.error}<br /><span class="err">{status?.error || live?.error}</span
					>{/if}
			</p>
			{#if status?.last_eval}
				<p class="meta">
					last eval ({status.last_eval.kind ?? '?'} @ epoch {status.last_eval.epoch ?? '?'}):
					<strong>{pct(status.last_eval.pass_rate)}</strong>
					({status.last_eval.n_pass ?? '?'}/{status.last_eval.n_total ?? '?'} cells,
					{status.last_eval.n_tiles ?? '?'} tiles)
				</p>
			{/if}
		{/if}
	</section>

	<section class="card">
		<h2>Epochs</h2>
		{#if !epochTail.length}
			<p class="muted">No epoch lines yet.</p>
		{:else}
			<div class="scroll">
				<table>
					<thead>
						<tr>
							<th>epoch</th>
							<th>time</th>
							<th>train loss</th>
							<th>val loss</th>
							<th>val kp</th>
							<th>val desc</th>
							<th>step</th>
						</tr>
					</thead>
					<tbody>
						{#each epochTail.slice().reverse() as row}
							<tr>
								<td>{row.epoch}</td>
								<td>{fmtS(row.epoch_s)}</td>
								<td>{fmtLoss(row.train_loss)}</td>
								<td>{fmtLoss(row.val_loss)}</td>
								<td>{fmtLoss(row.val_loss_kp)}</td>
								<td>{fmtLoss(row.val_loss_desc)}</td>
								<td>{row.step}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>

	<section class="card">
		<h2>Loss</h2>
		{#if !lossTail.length}
			<p class="muted">No loss lines yet (logged every N steps).</p>
		{:else}
			<p class="muted">{lossTail.length} steps (newest first)</p>
			<div class="scroll">
				<table>
					<thead>
						<tr><th>step</th><th>epoch</th><th>total</th><th>kp</th><th>desc</th></tr>
					</thead>
					<tbody>
						{#each lossTail.slice().reverse() as row}
							<tr>
								<td>{row.step}</td>
								<td>{row.epoch ?? '—'}</td>
								<td>{fmtLoss(row.loss_total)}</td>
								<td>{fmtLoss(row.loss_kp)}</td>
								<td>{fmtLoss(row.loss_desc)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>

	<section class="card">
		<h2>Eval</h2>
		{#if !evalTail.length}
			<p class="muted">No eval yet (val tiles × angles; test at end).</p>
		{:else}
			<div class="scroll">
				<table>
					<thead>
						<tr><th>kind</th><th>epoch</th><th>pass</th><th>cells</th><th>tiles</th></tr>
					</thead>
					<tbody>
						{#each evalTail.slice().reverse() as row}
							<tr>
								<td>{row.kind}</td>
								<td>{row.epoch}</td>
								<td>{pct(row.pass_rate as number)}</td>
								<td>{row.n_pass}/{row.n_total}</td>
								<td>{row.n_tiles ?? '—'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</section>
</div>

<style>
	.page {
		flex: 1;
		min-height: 0;
		overflow: auto;
		padding: 20px 24px 48px;
		max-width: 960px;
		width: 100%;
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
	.scroll {
		max-height: 420px;
		overflow: auto;
		margin-top: 8px;
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
