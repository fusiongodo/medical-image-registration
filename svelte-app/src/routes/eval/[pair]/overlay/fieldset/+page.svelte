<script lang="ts">
	import RegwsiOverlayViewer from '$lib/eval/RegwsiOverlayViewer.svelte';
	import type { FullMeta } from '$lib/server/evalOverlay';

	let { data } = $props();

	let phase = $state<'he' | 'warp'>('he');
	let generating = $state(false);
	let genError = $state<string | null>(null);
	let stage = $state<string | null>(null);
	let done = $state(0);
	let total = $state(0);
	let ready = $state(false);
	let heReady = $state(false);
	let fullMeta = $state<FullMeta | null>(null);
	let setLabel = $state<string | null>(null);
	let layer = $state('ihc_fieldset_tps');
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	const dataset = $derived(data.dataset ?? 'muromi');
	const subtitle = $derived(
		setLabel
			? `HE vs ${data.lam}/${data.estimator} · ${setLabel}`
			: `HE vs ${data.lam}/${data.estimator}`
	);

	function makeFullUrl() {
		const q = new URLSearchParams({
			pair: String(data.pairId),
			dataset,
			layers: 'he'
		});
		return `/api/eval/make-full?${q}`;
	}

	function statusUrl() {
		const q = new URLSearchParams({
			pair: String(data.pairId),
			estimator: data.estimator,
			lam: data.lam,
			dataset
		});
		if (data.batch) q.set('batch', data.batch);
		return `/api/eval/fieldset-full?${q}`;
	}

	function applyJob(job: { running?: boolean; stage?: string; done?: number; total?: number; error?: string | null } | null) {
		if (!job) return;
		generating = job.running === true;
		stage = job.stage ?? stage;
		done = job.done ?? 0;
		total = job.total ?? 0;
		if (job.error) genError = job.error;
	}

	async function loadMeta() {
		const q = new URLSearchParams({
			pair: String(data.pairId),
			dataset,
			meta: '1'
		});
		const r = await fetch(`/api/eval/full?${q}`);
		if (!r.ok) throw new Error(await r.text());
		fullMeta = (await r.json()) as FullMeta;
	}

	async function refreshMakeFull() {
		const r = await fetch(makeFullUrl());
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		heReady = j.ready === true;
		applyJob(j.job);
		return j;
	}

	async function refreshStatus() {
		const r = await fetch(statusUrl());
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		ready = j.ready === true;
		if (typeof j.layer === 'string') layer = j.layer;
		if (j.stamp?.set_name) setLabel = j.stamp.set_name;
		else if (j.stamp?.set_id) setLabel = j.stamp.set_id;
		applyJob(j.job);
		return j;
	}

	function stopPoll() {
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	}

	function startPolling(kind: 'he' | 'warp') {
		if (pollTimer) return;
		pollTimer = setInterval(async () => {
			try {
				if (kind === 'he') {
					const j = await refreshMakeFull();
					if (j.ready && !j.job?.running) {
						stopPoll();
						generating = false;
						await loadMeta();
						await startWarp();
					} else if (j.job && !j.job.running && j.job.error) {
						stopPoll();
						generating = false;
						genError = j.job.error;
					}
				} else {
					const j = await refreshStatus();
					if (j.ready && !j.job?.running) {
						stopPoll();
						generating = false;
					} else if (j.job && !j.job.running && j.job.error) {
						stopPoll();
						generating = false;
						genError = j.job.error;
					}
				}
			} catch (e) {
				stopPoll();
				generating = false;
				genError = e instanceof Error ? e.message : 'status failed';
			}
		}, 800);
	}

	async function startHe() {
		phase = 'he';
		genError = null;
		generating = true;
		const r = await fetch('/api/eval/make-full', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: data.pairId, dataset, layers: ['he'] })
		});
		if (!r.ok) {
			genError = await r.text();
			generating = false;
			return;
		}
		startPolling('he');
	}

	async function startWarp() {
		phase = 'warp';
		genError = null;
		generating = true;
		const r = await fetch('/api/eval/fieldset-full', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				pair_id: data.pairId,
				estimator: data.estimator,
				lam: data.lam,
				batch: data.batch,
				dataset
			})
		});
		if (!r.ok) {
			genError = await r.text();
			generating = false;
			return;
		}
		const j = await r.json();
		if (typeof j.layer === 'string') layer = j.layer;
		startPolling('warp');
	}

	function retry() {
		if (phase === 'he') void startHe();
		else void startWarp();
	}

	$effect(() => {
		void data.pairId;
		void data.estimator;
		void data.lam;
		void data.batch;
		void dataset;
		ready = false;
		heReady = false;
		genError = null;
		void (async () => {
			try {
				const he = await refreshMakeFull();
				if (he.ready) {
					await loadMeta();
					const j = await refreshStatus();
					if (!j.ready && !j.job?.running) await startWarp();
					else if (j.job?.running) {
						phase = 'warp';
						startPolling('warp');
					}
				} else if (he.job?.running) {
					phase = 'he';
					startPolling('he');
				} else {
					await startHe();
				}
			} catch (e) {
				genError = e instanceof Error ? e.message : 'init failed';
			}
		})();
		return () => stopPoll();
	});
</script>

{#if genError}
	<div class="empty err">
		Failed to build overlay: {genError}
		<button type="button" class="retry" onclick={retry}>Retry</button>
	</div>
{:else if !heReady || !fullMeta || generating || !ready}
	<div class="empty">
		{#if phase === 'he' || !heReady}
			Building HE mosaic…
		{:else}
			Generating field-set warp ({data.lam}/{data.estimator})…
		{/if}
		{#if total}
			<span class="prog">{done}/{total}{#if stage} · {stage}{/if}</span>
		{/if}
	</div>
{:else}
	<RegwsiOverlayViewer
		pairId={data.pairId}
		dataset={data.dataset}
		title={`Field overlay · pair ${data.pairId}`}
		{subtitle}
		movingLayer={layer}
		fullMeta={fullMeta}
		movingReady={true}
		batch={data.batch}
		lam={data.lam}
		estimator={data.estimator}
	/>
{/if}

<style>
	.empty {
		padding: 2rem;
		color: #9ca3af;
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.prog {
		display: block;
		margin-top: 0.5rem;
		font-variant-numeric: tabular-nums;
		color: #7dd3fc;
	}
	.err {
		color: #f87171;
	}
	.retry {
		margin-top: 0.75rem;
		padding: 0.35rem 0.75rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		background: #181b23;
		color: #e8eaf0;
		cursor: pointer;
	}
</style>
