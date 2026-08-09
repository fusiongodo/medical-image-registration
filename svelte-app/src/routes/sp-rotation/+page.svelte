<script lang="ts">
	import { goto } from '$app/navigation';
	import type { SpRotJobState } from '$lib/spRotJobs';

	type DatasetId = 'muromi' | 'acrobat';
	type PageData = {
		pairs: { pairId: number; ready: boolean }[];
		dataset: DatasetId;
		datasets: { id: DatasetId; label: string; pairCount: number }[];
	};

	type RunManifest = {
		id: string;
		name: string;
		dataset?: string;
		pairs: number[];
		angles: number[];
		created_at?: number;
		status?: {
			state?: string;
			done?: number;
			total?: number;
			detail?: string;
			error?: string | null;
			failed?: number;
			skipped?: number;
		};
	};

	type Cell = {
		pair_id: number;
		angle: number;
		state: string;
		label: string | null;
		n_inliers: number | null;
		rmse_px: number | null;
		rot_err_deg: number | null;
		trans_err_px: number | null;
		error: string | null;
	};

	let { data }: { data: PageData } = $props();

	let runs = $state<RunManifest[]>([]);
	let runId = $state<string | null>(null);
	let cells = $state<Cell[]>([]);
	let angles = $state<number[]>([]);
	let pairs = $state<number[]>([]);
	let showNew = $state(false);
	let newName = $state('');
	let newPairs = $state<number[]>([]);
	let createBusy = $state(false);
	let createErr = $state<string | null>(null);
	let runJob = $state<SpRotJobState | null>(null);
	let runPoll: ReturnType<typeof setInterval> | null = null;
	let matrixBusy = $state(false);
	let summary = $state<{
		by_angle?: Record<
			string,
			{ fail_rate: number | null; labeled: number; counts: Record<string, number> }
		>;
	} | null>(null);

	const dataset = $derived(data.dataset ?? 'muromi');
	const selectedRun = $derived(runs.find((r) => r.id === runId) ?? null);
	const batchRunning = $derived(
		!!runJob?.running || ['running', 'cell', 'gt'].includes((selectedRun?.status?.state || '').toLowerCase())
	);
	const progressDone = $derived(
		runJob?.running ? runJob.done : (selectedRun?.status?.done ?? runJob?.done ?? 0)
	);
	const progressTotal = $derived(
		runJob?.running ? runJob.total : (selectedRun?.status?.total ?? runJob?.total ?? 0)
	);
	const progressDetail = $derived(
		runJob?.running
			? runJob.detail
			: (selectedRun?.status?.detail || runJob?.detail || null)
	);

	async function loadRuns() {
		const r = await fetch('/api/sp-rotation/runs');
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		const all = Array.isArray(j.runs) ? j.runs : [];
		runs = all.filter((b: RunManifest) => (b.dataset || 'muromi') === dataset);
		if (!runId && runs.length) runId = runs[0].id;
		if (runId && !runs.some((b) => b.id === runId)) {
			runId = runs[0]?.id ?? null;
		}
	}

	async function loadMatrix() {
		if (!runId) {
			cells = [];
			angles = [];
			pairs = [];
			return;
		}
		matrixBusy = true;
		try {
			const r = await fetch(`/api/sp-rotation/runs/status?run=${encodeURIComponent(runId)}`);
			if (!r.ok) throw new Error(await r.text());
			const j = await r.json();
			cells = Array.isArray(j.cells) ? j.cells : [];
			angles = (j.manifest?.angles as number[]) || [];
			pairs = (j.manifest?.pairs as number[]) || [];
		} finally {
			matrixBusy = false;
		}
	}

	async function loadSummary() {
		if (!runId) {
			summary = null;
			return;
		}
		const r = await fetch(`/api/sp-rotation/runs/summary?run=${encodeURIComponent(runId)}`);
		if (!r.ok) {
			summary = null;
			return;
		}
		summary = await r.json();
	}

	function setDataset(next: DatasetId) {
		if (next === dataset) return;
		runId = null;
		void goto(`/sp-rotation?dataset=${next}`, { invalidateAll: true });
	}

	function cellAt(pid: number, ang: number): Cell | null {
		return cells.find((c) => c.pair_id === pid && c.angle === ang) ?? null;
	}

	function cellClass(c: Cell | null): string {
		if (!c || c.state === 'missing') return 'miss';
		if (c.state === 'error') return 'err';
		if (c.label === 'pass') return 'pass';
		if (c.label === 'fail') return 'fail';
		if (c.label === 'unsure') return 'unsure';
		return 'done';
	}

	function cellTitle(c: Cell | null): string {
		if (!c) return 'missing';
		const bits = [c.state];
		if (c.label) bits.push(`label=${c.label}`);
		if (c.n_inliers != null) bits.push(`inliers=${c.n_inliers}`);
		if (c.rot_err_deg != null) bits.push(`rotΔ=${c.rot_err_deg.toFixed(1)}°`);
		if (c.rmse_px != null) bits.push(`rmse=${c.rmse_px.toFixed(2)}`);
		if (c.error) bits.push(c.error);
		return bits.join(' · ');
	}

	function togglePair(pid: number) {
		if (newPairs.includes(pid)) newPairs = newPairs.filter((p) => p !== pid);
		else newPairs = [...newPairs, pid].sort((a, b) => a - b);
	}

	async function createRun() {
		createBusy = true;
		createErr = null;
		try {
			const r = await fetch('/api/sp-rotation/runs', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newName || `sp-rot-${Date.now()}`,
					pairs: newPairs,
					dataset
				})
			});
			if (!r.ok) throw new Error(await r.text());
			const j = await r.json();
			showNew = false;
			newName = '';
			newPairs = [];
			await loadRuns();
			runId = j.run?.id ?? runId;
			await loadMatrix();
		} catch (e) {
			createErr = e instanceof Error ? e.message : 'create failed';
		} finally {
			createBusy = false;
		}
	}

	function stopPoll() {
		if (runPoll) {
			clearInterval(runPoll);
			runPoll = null;
		}
	}

	async function pollProgress() {
		if (!runId) return;
		const r = await fetch(`/api/sp-rotation/runs/progress?run=${encodeURIComponent(runId)}`);
		if (!r.ok) return;
		runJob = await r.json();
		if (!runJob?.running) {
			stopPoll();
			await loadRuns();
			await loadMatrix();
			await loadSummary();
		}
	}

	async function startRun() {
		if (!runId) return;
		const r = await fetch('/api/sp-rotation/runs/run', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ run_id: runId })
		});
		if (!r.ok) {
			alert(await r.text());
			return;
		}
		const j = await r.json();
		runJob = j.state;
		stopPoll();
		runPoll = setInterval(() => void pollProgress(), 1500);
	}

	$effect(() => {
		void dataset;
		void loadRuns().then(async () => {
			await loadMatrix();
			await loadSummary();
		});
		return () => stopPoll();
	});

	$effect(() => {
		const id = runId;
		if (!id) return;
		void loadMatrix().then(() => loadSummary());
	});
</script>

<div class="page">
	<header>
		<div>
			<h1>SP rotation bench</h1>
			<p class="sub">12×30° SuperPoint+LightGlue vs regWSI step-1 rigid · human labels calibrate metrics</p>
		</div>
		<div class="ds">
			{#each data.datasets as d}
				<button class:active={dataset === d.id} onclick={() => setDataset(d.id)}>{d.label}</button>
			{/each}
		</div>
	</header>

	<section class="bar">
		<label>
			Run
			<select
				value={runId ?? ''}
				onchange={(e) => {
					runId = (e.currentTarget as HTMLSelectElement).value || null;
				}}
			>
				<option value="">—</option>
				{#each runs as r}
					<option value={r.id}>{r.name} ({r.id})</option>
				{/each}
			</select>
		</label>
		<button onclick={() => (showNew = !showNew)}>{showNew ? 'Cancel' : 'New run'}</button>
		<button disabled={!runId || batchRunning} onclick={() => void startRun()}>
			{batchRunning ? 'Running…' : 'Start'}
		</button>
		<button disabled={!runId || matrixBusy} onclick={() => void loadMatrix().then(() => loadSummary())}>
			Refresh
		</button>
		{#if batchRunning || progressTotal}
			<span class="prog">
				{progressDone}/{progressTotal}
				{#if progressDetail}<span class="muted"> · {progressDetail}</span>{/if}
				{#if runJob?.error}<span class="err"> · {runJob.error}</span>{/if}
			</span>
		{/if}
	</section>

	{#if showNew}
		<section class="new">
			<input placeholder="Run name" bind:value={newName} />
			<p class="muted">Angles fixed: 0…330 step 30. Pick pairs with regWSI DF (ready).</p>
			<div class="pair-grid">
				{#each data.pairs as p}
					<button
						class:on={newPairs.includes(p.pairId)}
						class:ready={p.ready}
						disabled={!p.ready}
						onclick={() => togglePair(p.pairId)}
					>
						{p.pairId}
					</button>
				{/each}
			</div>
			<div class="row">
				<button disabled={createBusy || !newPairs.length} onclick={() => void createRun()}>
					{createBusy ? 'Creating…' : `Create (${newPairs.length})`}
				</button>
				{#if createErr}<span class="err">{createErr}</span>{/if}
			</div>
		</section>
	{/if}

	{#if selectedRun}
		<section class="matrix-wrap">
			<table>
				<thead>
					<tr>
						<th>pair</th>
						{#each angles as a}
							<th>{a}°</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each pairs as pid}
						<tr>
							<td class="pid">{pid}</td>
							{#each angles as a}
								{@const c = cellAt(pid, a)}
								<td class={cellClass(c)} title={cellTitle(c)}>
									<a href={`/sp-rotation/${encodeURIComponent(runId!)}/cell?pair=${pid}&angle=${a}`}>
										{#if c?.label}
											{c.label[0].toUpperCase()}
										{:else if c?.state === 'done'}
											·
										{:else if c?.state === 'error'}
											!
										{:else}
											–
										{/if}
									</a>
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
			<p class="legend">
				<span class="swatch pass"></span> pass
				<span class="swatch fail"></span> fail
				<span class="swatch unsure"></span> unsure
				<span class="swatch done"></span> done unlabeled
				<span class="swatch miss"></span> missing
				<span class="swatch err"></span> error
			</p>
		</section>

		{#if summary?.by_angle}
			<section class="summary">
				<h2>Fail rate by angle (labels)</h2>
				<div class="sum-row">
					{#each angles as a}
						{@const row = summary.by_angle?.[String(a)]}
						<div class="sum-cell">
							<div class="ang">{a}°</div>
							<div class="val">
								{#if row?.fail_rate == null}
									—
								{:else}
									{(100 * row.fail_rate).toFixed(0)}%
								{/if}
							</div>
							<div class="muted n">{row?.labeled ?? 0} labeled</div>
						</div>
					{/each}
				</div>
			</section>
		{/if}
	{:else}
		<p class="muted empty">Create a run to start the 12-angle grid.</p>
	{/if}
</div>

<style>
	.page {
		padding: 20px 24px 40px;
		max-width: 1200px;
		overflow: auto;
		height: 100%;
	}
	header {
		display: flex;
		justify-content: space-between;
		gap: 16px;
		align-items: flex-start;
		margin-bottom: 16px;
	}
	h1 {
		font-size: 1.35rem;
		font-weight: 650;
	}
	.sub {
		color: #9ca3af;
		font-size: 0.85rem;
		margin-top: 4px;
	}
	.ds {
		display: flex;
		gap: 6px;
	}
	.ds button,
	.bar button,
	.new button,
	.pair-grid button {
		all: unset;
		cursor: pointer;
		padding: 6px 10px;
		border-radius: 6px;
		background: #232733;
		color: #cfd3dc;
		font-size: 0.85rem;
		border: 1px solid #2f3340;
	}
	.ds button.active,
	.pair-grid button.on {
		background: #3a4663;
		border-color: #5b6b8f;
		color: #fff;
	}
	.pair-grid button:disabled {
		opacity: 0.35;
		cursor: not-allowed;
	}
	.bar {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		align-items: center;
		margin-bottom: 14px;
	}
	.bar label {
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 0.85rem;
		color: #9ca3af;
	}
	select,
	input {
		background: #181b23;
		border: 1px solid #2f3340;
		color: #e8eaf0;
		border-radius: 6px;
		padding: 6px 8px;
		font-size: 0.85rem;
	}
	.prog {
		font-size: 0.85rem;
	}
	.muted {
		color: #9ca3af;
	}
	.err {
		color: #f87171;
	}
	.new {
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 8px;
		padding: 12px;
		margin-bottom: 14px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.pair-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.pair-grid button.ready {
		border-color: #3f4d3a;
	}
	.row {
		display: flex;
		gap: 10px;
		align-items: center;
	}
	.matrix-wrap {
		overflow: auto;
		margin-bottom: 18px;
	}
	table {
		border-collapse: collapse;
		font-size: 0.8rem;
	}
	th,
	td {
		border: 1px solid #2a2d3a;
		width: 36px;
		height: 28px;
		text-align: center;
		padding: 0;
	}
	th {
		color: #9ca3af;
		font-weight: 500;
		background: #181b23;
	}
	.pid {
		padding: 0 8px;
		color: #cfd3dc;
		font-variant-numeric: tabular-nums;
	}
	td a {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		height: 100%;
		color: inherit;
		text-decoration: none;
	}
	td.pass {
		background: #1f3d2a;
		color: #86efac;
	}
	td.fail {
		background: #3f1d1d;
		color: #fca5a5;
	}
	td.unsure {
		background: #3a3420;
		color: #fde68a;
	}
	td.done {
		background: #1e2433;
		color: #93c5fd;
	}
	td.miss {
		background: #14161c;
		color: #4b5563;
	}
	td.err {
		background: #451a1a;
		color: #fb7185;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		align-items: center;
		margin-top: 10px;
		font-size: 0.75rem;
		color: #9ca3af;
	}
	.swatch {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 2px;
		margin-right: 4px;
	}
	.swatch.pass {
		background: #1f3d2a;
	}
	.swatch.fail {
		background: #3f1d1d;
	}
	.swatch.unsure {
		background: #3a3420;
	}
	.swatch.done {
		background: #1e2433;
	}
	.swatch.miss {
		background: #14161c;
		border: 1px solid #2a2d3a;
	}
	.swatch.err {
		background: #451a1a;
	}
	.summary h2 {
		font-size: 0.95rem;
		margin-bottom: 8px;
	}
	.sum-row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.sum-cell {
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 6px;
		padding: 8px 10px;
		min-width: 64px;
		text-align: center;
	}
	.ang {
		font-size: 0.75rem;
		color: #9ca3af;
	}
	.val {
		font-size: 1.05rem;
		font-variant-numeric: tabular-nums;
	}
	.n {
		font-size: 0.7rem;
	}
	.empty {
		margin-top: 24px;
	}
</style>
