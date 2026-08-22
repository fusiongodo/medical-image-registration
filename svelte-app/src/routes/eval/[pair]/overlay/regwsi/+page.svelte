<script lang="ts">
	import RegwsiOverlayViewer from '$lib/eval/RegwsiOverlayViewer.svelte';
	import type { FullMeta } from '$lib/server/evalOverlay';

	let { data } = $props();

	let generating = $state(false);
	let genError = $state<string | null>(null);
	let stage = $state<string | null>(null);
	let done = $state(0);
	let total = $state(0);
	let ready = $state(false);
	let fullMeta = $state<FullMeta | null>(null);
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	const dataset = $derived(data.dataset ?? 'muromi');
	const layers = ['he', 'ihc_warped'];
	let pollLayers = $state<string[]>(layers);

	function statusUrl(ls: string[] = pollLayers) {
		const q = new URLSearchParams({
			pair: String(data.pairId),
			dataset,
			layers: ls.join(',')
		});
		return `/api/eval/make-full?${q}`;
	}

	async function layerStatus(ls: string[]) {
		const r = await fetch(statusUrl(ls));
		if (!r.ok) throw new Error(await r.text());
		return r.json();
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

	async function refreshStatus() {
		const j = await layerStatus(pollLayers);
		const all = await layerStatus(layers);
		ready = all.ready === true;
		if (j.job) {
			generating = j.job.running === true;
			stage = j.job.stage;
			done = j.job.done ?? 0;
			total = j.job.total ?? 0;
			if (j.job.error) genError = j.job.error;
		}
		return { ...j, ready: all.ready === true };
	}

	function stopPoll() {
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	}

	function startPolling() {
		if (pollTimer) return;
		pollTimer = setInterval(async () => {
			try {
				const j = await refreshStatus();
				if (j.ready && !j.job?.running) {
					stopPoll();
					generating = false;
					await loadMeta();
				} else if (j.job && !j.job.running && j.job.error) {
					stopPoll();
					generating = false;
					genError = j.job.error;
				}
			} catch (e) {
				stopPoll();
				generating = false;
				genError = e instanceof Error ? e.message : 'status failed';
			}
		}, 800);
	}

	async function startGenerate(force = false) {
		genError = null;
		generating = true;
		ready = false;
		const he = await layerStatus(['he']);
		const warped = await layerStatus(['ihc_warped']);
		const need: string[] = [];
		if (!he.ready) need.push('he');
		if (force || !warped.ready) need.push('ihc_warped');
		if (!need.length) {
			generating = false;
			ready = true;
			await loadMeta();
			return;
		}
		pollLayers = need;
		const r = await fetch('/api/eval/make-full', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: data.pairId, dataset, layers: need, force })
		});
		if (!r.ok) {
			genError = await r.text();
			generating = false;
			return;
		}
		startPolling();
	}

	$effect(() => {
		void data.pairId;
		void dataset;
		ready = false;
		genError = null;
		void (async () => {
			try {
				const j = await refreshStatus();
				if (j.ready) {
					await loadMeta();
				} else if (j.job?.running) {
					startPolling();
				} else {
					await startGenerate();
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
		<button type="button" class="retry" onclick={() => startGenerate(true)}>Retry</button>
	</div>
{:else if !ready || !fullMeta || generating}
	<div class="empty">
		Building overlay mosaics…
		{#if total}
			<span class="prog">{done}/{total}{#if stage} · {stage}{/if}</span>
		{/if}
	</div>
{:else}
	<RegwsiOverlayViewer
		pairId={data.pairId}
		dataset={data.dataset}
		title={`regWSI overlay · pair ${data.pairId}`}
		subtitle="HE vs DeeperHistReg warped IHC"
		movingLayer="ihc_warped"
		fullMeta={fullMeta}
		movingReady={true}
		batch={data.batch}
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
