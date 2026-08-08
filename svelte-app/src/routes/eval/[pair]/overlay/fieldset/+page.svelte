<script lang="ts">
	import RegwsiOverlayViewer from '$lib/eval/RegwsiOverlayViewer.svelte';

	let { data } = $props();

	let generating = $state(false);
	let genError = $state<string | null>(null);
	let stage = $state<string | null>(null);
	let done = $state(0);
	let total = $state(0);
	let ready = $state(false);
	let setLabel = $state<string | null>(null);
	let layer = $state('ihc_fieldset_tps');
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	const subtitle = $derived(
		setLabel
			? `HE vs ${data.lam}/${data.estimator} · ${setLabel}`
			: `HE vs ${data.lam}/${data.estimator}`
	);

	function statusUrl() {
		const q = new URLSearchParams({
			pair: String(data.pairId),
			estimator: data.estimator,
			lam: data.lam,
			dataset: data.dataset ?? 'muromi'
		});
		if (data.batch) q.set('batch', data.batch);
		return `/api/eval/fieldset-full?${q}`;
	}

	async function refreshStatus() {
		const r = await fetch(statusUrl());
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		ready = j.ready === true;
		if (typeof j.layer === 'string') layer = j.layer;
		if (j.stamp?.set_name) setLabel = j.stamp.set_name;
		else if (j.stamp?.set_id) setLabel = j.stamp.set_id;
		if (j.job) {
			generating = j.job.running === true;
			stage = j.job.stage;
			done = j.job.done ?? 0;
			total = j.job.total ?? 0;
			if (j.job.error) genError = j.job.error;
		}
		return j;
	}

	async function startGenerate() {
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
				dataset: data.dataset ?? 'muromi'
			})
		});
		if (!r.ok) {
			genError = await r.text();
			generating = false;
			return;
		}
		const j = await r.json();
		if (typeof j.layer === 'string') layer = j.layer;
		startPolling();
	}

	function startPolling() {
		if (pollTimer) return;
		pollTimer = setInterval(async () => {
			try {
				const j = await refreshStatus();
				if (j.ready && !j.job?.running) {
					generating = false;
					if (pollTimer) {
						clearInterval(pollTimer);
						pollTimer = null;
					}
				} else if (j.job && !j.job.running && j.job.error) {
					generating = false;
					genError = j.job.error;
					if (pollTimer) {
						clearInterval(pollTimer);
						pollTimer = null;
					}
				}
			} catch (e) {
				genError = e instanceof Error ? e.message : 'status failed';
				generating = false;
				if (pollTimer) {
					clearInterval(pollTimer);
					pollTimer = null;
				}
			}
		}, 800);
	}

	$effect(() => {
		void data.pairId;
		void data.estimator;
		void data.lam;
		void data.batch;
		ready = false;
		void (async () => {
			try {
				const j = await refreshStatus();
				if (!j.ready && !j.job?.running) await startGenerate();
				else if (j.job?.running) startPolling();
			} catch (e) {
				genError = e instanceof Error ? e.message : 'init failed';
			}
		})();
		return () => {
			if (pollTimer) {
				clearInterval(pollTimer);
				pollTimer = null;
			}
		};
	});
</script>

{#if !data.heReady || !data.fullMeta}
	<div class="empty">
		No HE mosaic for pair {data.pairId}. Run
		<code>python regWSI/make_full.py {data.pairId} --layers he ihc</code>
	</div>
{:else if generating || (!ready && !genError)}
	<div class="empty">
		Generating field-set warp ({data.lam}/{data.estimator})…
		{#if total}
			<span class="prog">{done}/{total}{#if stage} · {stage}{/if}</span>
		{/if}
	</div>
{:else if genError}
	<div class="empty err">
		Failed to build field-set mosaic: {genError}
		<button type="button" class="retry" onclick={() => startGenerate()}>Retry</button>
	</div>
{:else}
	<RegwsiOverlayViewer
		pairId={data.pairId}
		dataset={data.dataset}
		title={`Field overlay · pair ${data.pairId}`}
		{subtitle}
		movingLayer={layer}
		fullMeta={data.fullMeta}
		movingReady={true}
	/>
{/if}

<style>
	.empty {
		padding: 2rem;
		color: #9ca3af;
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.empty code {
		display: block;
		margin-top: 0.5rem;
		padding: 0.5rem 0.65rem;
		background: #181b23;
		border-radius: 3px;
		font-size: 0.8rem;
		color: #c4c9d4;
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
