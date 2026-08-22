<script lang="ts">
	import { goto } from '$app/navigation';
	import EvalEpsSweep from '$lib/eval/EvalEpsSweep.svelte';
	import EvalTreMatrix, { type BatchTreResult } from '$lib/eval/EvalTreMatrix.svelte';
	import type { BatchJobState } from '$lib/evalJobs';

	type DatasetId = 'muromi' | 'acrobat' | 'anhir';
	type EvalPageData = {
		pairs: {
			pairId: number;
			ready: boolean;
			mosaicReady: boolean;
			landmarkCount: number;
			mainSetId: string | null;
			mainSetName: string | null;
		}[];
		dataset: DatasetId;
		datasets: { id: DatasetId; label: string; pairCount: number }[];
	};

	let { data }: { data: EvalPageData } = $props();

	interface BatchManifest {
		id: string;
		name: string;
		dataset?: string;
		pairs: number[];
		lams?: string[];
		estimators?: string[];
		config?: Record<string, unknown>;
		status?: { state?: string; done?: number; total?: number; detail?: string; error?: string | null };
	}

	let batches = $state<BatchManifest[]>([]);
	let batchId = $state<string | null>(null);
	let selectedPair = $state<number | null>(null);
	let tre = $state<BatchTreResult | null>(null);
	let treBusy = $state(false);
	let treErr = $state<string | null>(null);
	let pairTres = $state<
		Record<string, { regwsi: number | null; fft: number | null; superpoint_glue: number | null }>
	>({});
	let fetchGen = 0;
	type TreKey = 'regwsi' | 'fft' | 'superpoint_glue';
	let sortKey = $state<TreKey | 'pair'>('pair');
	let sortDir = $state<'asc' | 'desc'>('asc');

	let showNew = $state(false);
	let newName = $state('');
	let newPairs = $state<number[]>([]);
	let newWendland = $state(0.35);
	let newBsplineGrid = $state(8);
	let newBsplineReg = $state(0.001);
	let newForce = $state(false);
	let createBusy = $state(false);
	let createErr = $state<string | null>(null);

	let runJob = $state<BatchJobState | null>(null);
	let runPoll: ReturnType<typeof setInterval> | null = null;

	const dataset = $derived(data.dataset ?? 'muromi');
	const selectedBatch = $derived(batches.find((b) => b.id === batchId) ?? null);
	const batchRunning = $derived(
		!!runJob?.running || (selectedBatch?.status?.state || '').toLowerCase() === 'running'
	);
	const progressDone = $derived(
		runJob?.running ? runJob.done : (selectedBatch?.status?.done ?? runJob?.done ?? 0)
	);
	const progressTotal = $derived(
		runJob?.running ? runJob.total : (selectedBatch?.status?.total ?? runJob?.total ?? 0)
	);
	const progressDetail = $derived(
		runJob?.running
			? runJob.detail
			: (selectedBatch?.status?.detail || runJob?.detail || null)
	);
	const listPairs = $derived.by(() => {
		let rows;
		if (selectedBatch?.pairs?.length) {
			const set = new Set(selectedBatch.pairs);
			rows = data.pairs.filter((p) => set.has(p.pairId));
		} else if (dataset !== 'muromi') {
			rows = data.pairs;
		} else {
			rows = data.pairs.filter((p) => p.landmarkCount > 0);
		}
		const key = sortKey;
		const dir = sortDir === 'asc' ? 1 : -1;
		return [...rows].sort((a, b) => {
			if (key === 'pair') return (a.pairId - b.pairId) * dir;
			const va = pairTres[String(a.pairId)]?.[key];
			const vb = pairTres[String(b.pairId)]?.[key];
			const na = va == null || !Number.isFinite(va);
			const nb = vb == null || !Number.isFinite(vb);
			if (na && nb) return a.pairId - b.pairId;
			if (na) return 1;
			if (nb) return -1;
			const d = (va as number) - (vb as number);
			return d === 0 ? a.pairId - b.pairId : d * dir;
		});
	});
	const selected = $derived(
		selectedPair == null ? null : (data.pairs.find((p) => p.pairId === selectedPair) ?? null)
	);
	const selectablePairs = $derived(
		dataset !== 'muromi' ? data.pairs : data.pairs.filter((p) => p.landmarkCount > 0)
	);

	async function loadBatches() {
		const r = await fetch('/api/eval/batches');
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		const all = Array.isArray(j.batches) ? j.batches : [];
		batches = all.filter((b: BatchManifest) => (b.dataset || 'muromi') === dataset);
		if (!batchId && batches.length) batchId = batches[0].id;
		if (batchId && !batches.some((b) => b.id === batchId)) {
			batchId = batches[0]?.id ?? null;
		}
	}

	function setDataset(next: DatasetId) {
		if (next === dataset) return;
		batchId = null;
		selectedPair = null;
		tre = null;
		void goto(`/eval?dataset=${next}`, { invalidateAll: true });
	}

	async function fetchTre(pairId: number) {
		if (!batchId) {
			tre = null;
			treErr = 'Select or create a batch';
			return;
		}
		const gen = ++fetchGen;
		treBusy = true;
		treErr = null;
		tre = null;
		try {
			const r = await fetch(`/api/eval/tre?pair=${pairId}&batch=${encodeURIComponent(batchId)}`);
			if (!r.ok) throw new Error(await r.text());
			const json = await r.json();
			if (gen !== fetchGen) return;
			if (json.error) {
				treErr = json.error;
				tre = null;
			} else {
				tre = json;
				if (batchId) void loadPairTres(batchId);
			}
		} catch (e) {
			if (gen !== fetchGen) return;
			treErr = e instanceof Error ? e.message : 'tre failed';
			tre = null;
		} finally {
			if (gen === fetchGen) treBusy = false;
		}
	}

	function selectPair(pairId: number) {
		if (selectedPair === pairId) {
			selectedPair = null;
			tre = null;
			treErr = null;
			fetchGen += 1;
			return;
		}
		selectedPair = pairId;
		void fetchTre(pairId);
	}

	function closeTre() {
		selectedPair = null;
		tre = null;
		treErr = null;
		treBusy = false;
		fetchGen += 1;
	}

	function openNewModal() {
		const d = new Date();
		const pad = (n: number) => String(n).padStart(2, '0');
		newName = `run-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
		newPairs = selectablePairs.map((p) => p.pairId);
		createErr = null;
		showNew = true;
	}

	function toggleNewPair(id: number) {
		if (newPairs.includes(id)) newPairs = newPairs.filter((p) => p !== id);
		else newPairs = [...newPairs, id].sort((a, b) => a - b);
	}

	async function createBatch() {
		createBusy = true;
		createErr = null;
		try {
			const r = await fetch('/api/eval/batches', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newName,
					dataset,
					pairs: newPairs,
					config: {
						wendland_eps: newWendland,
						bspline_grid: newBsplineGrid,
						bspline_reg: newBsplineReg,
						force: newForce
					}
				})
			});
			if (!r.ok) {
				const t = await r.text();
				let msg = t;
				try {
					msg = JSON.parse(t).message || t;
				} catch {
					/* keep raw */
				}
				throw new Error(msg);
			}
			const j = await r.json();
			await loadBatches();
			batchId = j.manifest?.id ?? batchId;
			showNew = false;
		} catch (e) {
			createErr = e instanceof Error ? e.message : 'create failed';
		} finally {
			createBusy = false;
		}
	}

	async function runBatch() {
		if (!batchId) return;
		const r = await fetch('/api/eval/batches/run', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ batch_id: batchId })
		});
		if (!r.ok) {
			treErr = await r.text();
			return;
		}
		const j = await r.json();
		runJob = j.state;
		startRunPoll();
	}

	function stopRunPoll() {
		if (runPoll) {
			clearInterval(runPoll);
			runPoll = null;
		}
	}

	function startRunPoll() {
		if (runPoll || !batchId) return;
		runPoll = setInterval(async () => {
			if (!batchId) return;
			const r = await fetch(`/api/eval/batches/progress?batch=${encodeURIComponent(batchId)}`);
			const j = r.ok ? await r.json() : null;
			runJob = j;
			if (j && !j.running) {
				stopRunPoll();
				await loadBatches();
				if (batchId) void loadPairTres(batchId);
				if (selectedPair != null) void fetchTre(selectedPair);
			} else if (j?.running) {
				const b = batches.find((x) => x.id === batchId);
				if (b) {
					b.status = {
						state: 'running',
						done: j.done,
						total: j.total,
						detail: j.detail ?? '',
						error: j.error
					};
					batches = [...batches];
				}
			}
		}, 1000);
	}

	$effect(() => {
		void dataset;
		void loadBatches().catch((e) => {
			treErr = e instanceof Error ? e.message : 'failed to load batches';
		});
		return () => {
			stopRunPoll();
		};
	});

	$effect(() => {
		const id = batchId;
		stopRunPoll();
		runJob = null;
		if (!id) return;
		let cancelled = false;
		void fetch(`/api/eval/batches/progress?batch=${encodeURIComponent(id)}`)
			.then((r) => (r.ok ? r.json() : null))
			.then((j) => {
				if (cancelled || !j) return;
				runJob = j;
				if (j.running) startRunPoll();
			});
		return () => {
			cancelled = true;
		};
	});

	$effect(() => {
		void batchId;
		if (selectedPair != null) void fetchTre(selectedPair);
	});

	$effect(() => {
		const id = batchId;
		pairTres = {};
		if (!id) return;
		let cancelled = false;
		void loadPairTres(id).catch(() => {
			if (!cancelled) pairTres = {};
		});
		return () => {
			cancelled = true;
		};
	});

	async function loadPairTres(id: string) {
		const r = await fetch(`/api/eval/pair-tres?batch=${encodeURIComponent(id)}`);
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		if (id !== batchId) return;
		pairTres = j.pairs ?? {};
	}

	function fmtTre(v: number | null | undefined) {
		return v == null ? '—' : v.toFixed(2);
	}

	function setSort(key: TreKey | 'pair') {
		if (sortKey === key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
		else {
			sortKey = key;
			sortDir = 'asc';
		}
	}

	function sortMark(key: TreKey | 'pair') {
		if (sortKey !== key) return '';
		return sortDir === 'asc' ? ' ↑' : ' ↓';
	}
</script>

<div class="page">
	<div class="batch-bar">
		<label>
			Dataset
			<select
				value={dataset}
				onchange={(e) => setDataset((e.currentTarget as HTMLSelectElement).value as DatasetId)}
			>
				{#each data.datasets as d}
					<option value={d.id}>{d.label} ({d.pairCount})</option>
				{/each}
			</select>
		</label>
		<label>
			Batch
			<select
				value={batchId ?? ''}
				onchange={(e) => {
					const v = (e.currentTarget as HTMLSelectElement).value;
					batchId = v || null;
				}}
			>
				<option value="">—</option>
				{#each batches as b}
					<option value={b.id}>{b.name} ({b.id})</option>
				{/each}
			</select>
		</label>
		<button type="button" class="btn" onclick={openNewModal}>New batch</button>
		<button type="button" class="btn primary" disabled={!batchId || batchRunning} onclick={runBatch}>
			{batchRunning ? 'Running…' : 'Run batch'}
		</button>
		{#if batchRunning}
			<span class="prog">
				{progressDone}/{progressTotal || '?'}{#if progressDetail}
					· {progressDetail}{/if}
			</span>
		{:else if runJob?.error || selectedBatch?.status?.error}
			<span class="err">{runJob?.error || selectedBatch?.status?.error}</span>
		{:else if selectedBatch?.status?.state}
			<span class="prog muted">
				status: {selectedBatch.status.state}{#if (selectedBatch.status.done ?? 0) > 0 || (selectedBatch.status.total ?? 0) > 0}
					· {selectedBatch.status.done ?? 0}/{selectedBatch.status.total || '?'}{/if}
			</span>
		{/if}
		<a class="link" href={`/eval/0/annotate?dataset=${dataset}`}>Annotate landmarks</a>
	</div>

	<div class="layout" class:with-panel={selectedPair != null}>
		<div class="list-col">
			<header>
				<h1>Evaluation</h1>
				<p class="sub">
					{#if dataset === 'acrobat'}
						ACROBAT: regWSI first (DF + rigid), then LAM × field. Official TRE via Grand Challenge
						upload.
					{:else if dataset === 'anhir'}
						ANHIR training: local TRE from both-side landmarks. Canvas TIFFs under data/anhir/regwsi.
					{:else}
						muROMI: batch TRE for LAM × field. Landmarks under data/regwsi; methods under
						data/eval_runs.
					{/if}
				</p>
			</header>
			{#if dataset === 'anhir'}
				<EvalEpsSweep {dataset} />
			{/if}
			<div class="list-head row">
				<div class="main">
					<button type="button" class="sort" class:on={sortKey === 'pair'} onclick={() => setSort('pair')}>
						Pair{sortMark('pair')}
					</button>
					<span class="tres">
						<button type="button" class="sort" class:on={sortKey === 'regwsi'} onclick={() => setSort('regwsi')}
							>reg{sortMark('regwsi')}</button
						>
						<button type="button" class="sort" class:on={sortKey === 'fft'} onclick={() => setSort('fft')}
							>fft{sortMark('fft')}</button
						>
						<button
							type="button"
							class="sort"
							class:on={sortKey === 'superpoint_glue'}
							onclick={() => setSort('superpoint_glue')}>sp{sortMark('superpoint_glue')}</button
						>
					</span>
					<span class="meta"></span>
				</div>
				<span class="action ghost">Annotate</span>
				<span class="action ghost">Overlay</span>
			</div>
			<ul class="list">
				{#each listPairs as p}
					<li class:ready={p.ready} class:selected={selectedPair === p.pairId}>
						<div class="row">
							<button type="button" class="main" onclick={() => selectPair(p.pairId)}>
								<span class="label">Pair {p.pairId}</span>
								{#if batchId}
									{@const t = pairTres[String(p.pairId)]}
									<span class="tres" title="regWSI / FFT / SP+LG">
										<span>{fmtTre(t?.regwsi)}</span>
										<span>{fmtTre(t?.fft)}</span>
										<span>{fmtTre(t?.superpoint_glue)}</span>
									</span>
								{:else}
									<span class="tres"></span>
								{/if}
								<span class="meta">
									{#if p.landmarkCount > 0}
										<span class="landmarks">{p.landmarkCount} landmarks</span>
									{/if}
									<span class="badge" class:ok={p.ready}>{p.ready ? 'regWSI' : 'no regWSI'}</span>
									<span class="badge" class:ok={p.mosaicReady} title={p.mosaicReady ? 'Overlay mosaic ready' : 'Overlay mosaic not built yet'}
										>{p.mosaicReady ? 'mosaic' : 'no mosaic'}</span
									>
								</span>
							</button>
							<a class="action" href={`/eval/${p.pairId}/annotate?dataset=${dataset}`}>Annotate</a>
							<a
								class="action"
								href={`/eval/${p.pairId}/overlay/regwsi?${new URLSearchParams({
									dataset,
									...(batchId ? { batch: batchId } : {})
								})}`}>Overlay</a
							>
						</div>
					</li>
				{/each}
			</ul>
		</div>
		{#if selected && selectedPair != null}
			<div class="panel-col">
				<EvalTreMatrix
					pairId={selectedPair}
					{dataset}
					{tre}
					{treBusy}
					{treErr}
					landmarkCount={selected.landmarkCount}
					{batchId}
					onClose={closeTre}
				/>
			</div>
		{/if}
	</div>
</div>

{#if showNew}
	<div class="modal-backdrop">
		<div class="modal" role="dialog" aria-modal="true">
			<h3>New eval batch</h3>
			<label>
				Name
				<input bind:value={newName} />
			</label>
			<p class="gate-note">Dataset: {dataset}</p>
			<div class="pair-pick">
				<span>{dataset === 'muromi' ? 'Pairs with landmarks' : `${dataset} pairs`}</span>
				<div class="chips">
					{#each selectablePairs as p}
						<button
							type="button"
							class="chip"
							class:on={newPairs.includes(p.pairId)}
							onclick={() => toggleNewPair(p.pairId)}
						>
							{p.pairId}
						</button>
					{/each}
				</div>
			</div>
			<p class="gate-note">Gate: keep 100% at L0–L3 · exclude 5% at L4 · exclude 10% at L5</p>
			<div class="cfg">
				<label>Wendland ε <input type="number" step="0.01" bind:value={newWendland} /></label>
				<label>B-spline grid <input type="number" step="1" bind:value={newBsplineGrid} /></label>
				<label>B-spline reg <input type="number" step="0.0001" bind:value={newBsplineReg} /></label>
				<label class="check"><input type="checkbox" bind:checked={newForce} /> force recompute</label>
			</div>
			{#if createErr}<p class="err">{createErr}</p>{/if}
			<div class="modal-actions">
				<button type="button" class="btn" onclick={() => (showNew = false)} disabled={createBusy}>Cancel</button>
				<button
					type="button"
					class="btn primary"
					onclick={createBatch}
					disabled={createBusy || !newName || !newPairs.length}
				>
					{createBusy ? 'Creating…' : 'Create'}
				</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.page {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}
	.batch-bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.6rem;
		padding: 0.65rem 1rem;
		border-bottom: 1px solid #2a2d3a;
		background: #0f1117;
	}
	.batch-bar label {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.78rem;
		color: #9ca3af;
	}
	.batch-bar select,
	.modal input {
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		color: #e8eaf0;
		padding: 0.3rem 0.45rem;
		font-size: 0.8rem;
	}
	.btn {
		padding: 0.35rem 0.7rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		background: #181b23;
		color: #e8eaf0;
		font-size: 0.78rem;
		cursor: pointer;
	}
	.btn.primary {
		border-color: #2563eb;
		background: #1d4ed8;
	}
	.btn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.prog {
		font-size: 0.75rem;
		color: #f59e0b;
	}
	.prog.muted {
		color: #6b7280;
	}
	.err {
		font-size: 0.75rem;
		color: #f87171;
	}
	.link {
		margin-left: auto;
		font-size: 0.75rem;
		color: #93c5fd;
		text-decoration: none;
	}
	.layout {
		flex: 1;
		min-height: 0;
		display: grid;
		grid-template-columns: 1fr;
	}
	.layout.with-panel {
		grid-template-columns: minmax(280px, 1fr) minmax(380px, 1.4fr);
	}
	.list-col {
		overflow: auto;
		padding: 1rem 1.25rem;
	}
	.panel-col {
		min-height: 0;
		overflow: hidden;
	}
	header h1 {
		margin: 0 0 0.35rem;
		font-size: 1.15rem;
		color: #e8eaf0;
	}
	.sub {
		margin: 0 0 1rem;
		font-size: 0.8rem;
		color: #9ca3af;
		max-width: 40rem;
	}
	.list-head {
		margin-bottom: 0.2rem;
	}
	.list-head .main {
		padding: 0.15rem 0.75rem;
		border: none;
		background: transparent;
		cursor: default;
	}
	.list-head .sort {
		padding: 0;
		border: none;
		background: none;
		color: #6b7280;
		font-size: 0.68rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		cursor: pointer;
		text-align: right;
	}
	.list-head .sort.on {
		color: #93c5fd;
	}
	.list-head .main > .sort {
		text-align: left;
	}
	.action.ghost {
		visibility: hidden;
		pointer-events: none;
	}
	.list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}
	.row {
		display: flex;
		align-items: stretch;
		gap: 0.35rem;
	}
	.main {
		flex: 1;
		display: grid;
		grid-template-columns: 7rem auto 1fr;
		align-items: center;
		gap: 0.75rem;
		padding: 0.55rem 0.75rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		background: #12141a;
		color: inherit;
		text-align: left;
		cursor: pointer;
	}
	li.selected .main {
		border-color: #3b82f6;
	}
	.label {
		font-weight: 600;
		color: #e8eaf0;
	}
	.meta {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		justify-self: end;
		font-size: 0.72rem;
		color: #9ca3af;
	}
	.tres {
		display: grid;
		grid-template-columns: 6.5ch 6.5ch 6.5ch;
		column-gap: 0.7rem;
		justify-self: start;
		font-variant-numeric: tabular-nums;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		color: #e8eaf0;
		min-height: 1em;
	}
	.tres span {
		text-align: right;
	}
	.badge {
		padding: 0.1rem 0.35rem;
		border-radius: 3px;
		background: #1f2330;
		color: #6b7280;
	}
	.badge.ok {
		color: #86efac;
	}
	.action {
		display: flex;
		align-items: center;
		padding: 0 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		font-size: 0.72rem;
		color: #93c5fd;
		text-decoration: none;
	}
	.modal-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.55);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 50;
	}
	.modal {
		width: min(520px, 92vw);
		background: #12141a;
		border: 1px solid #2a2d3a;
		border-radius: 6px;
		padding: 1rem 1.1rem;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.modal h3 {
		margin: 0;
		font-size: 0.95rem;
		color: #e8eaf0;
	}
	.modal label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.75rem;
		color: #9ca3af;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		margin-top: 0.35rem;
	}
	.chip {
		padding: 0.25rem 0.5rem;
		border: 1px solid #2a2d3a;
		border-radius: 999px;
		background: #181b23;
		color: #9ca3af;
		font-size: 0.72rem;
		cursor: pointer;
	}
	.chip.on {
		border-color: #3b82f6;
		color: #e8eaf0;
	}
	.gate-note {
		margin: 0;
		font-size: 0.72rem;
		color: #9ca3af;
	}
	.cfg {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem;
	}
	.check {
		flex-direction: row !important;
		align-items: center;
		gap: 0.4rem !important;
	}
	.modal-actions {
		display: flex;
		justify-content: flex-end;
		gap: 0.5rem;
	}
</style>
