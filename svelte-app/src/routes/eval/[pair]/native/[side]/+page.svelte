<script lang="ts">
	import { onMount } from 'svelte';
	import NativeSlideViewer from '$lib/eval/NativeSlideViewer.svelte';
	import type { FullMeta } from '$lib/server/evalOverlay';

	let { data } = $props();

	let generating = $state(false);
	let genError = $state<string | null>(null);
	let stage = $state<string | null>('checking');
	let done = $state(0);
	let total = $state(0);
	let ready = $state(false);
	let fullMeta = $state<FullMeta | null>(null);

	const dataset = $derived(data.dataset ?? 'muromi');
	const side = $derived(data.side as 'he' | 'ihc');

	function statusUrl() {
		const q = new URLSearchParams({
			pair: String(data.pairId),
			dataset,
			layers: side
		});
		return `/api/eval/make-full?${q}`;
	}

	async function layerStatus() {
		const r = await fetch(statusUrl());
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
		ready = true;
		generating = false;
		stage = 'done';
	}

	function applyJob(job: {
		running?: boolean;
		stage?: string | null;
		done?: number;
		total?: number;
		error?: string | null;
	} | null) {
		if (!job) return;
		generating = job.running === true;
		stage = job.stage ?? stage;
		done = job.done ?? done;
		total = job.total ?? total;
		if (job.error) genError = job.error;
	}

	async function waitUntilReady() {
		for (let i = 0; i < 600; i++) {
			const j = await layerStatus();
			applyJob(j.job);
			if (j.ready && !j.job?.running) {
				await loadMeta();
				return;
			}
			if (j.job && !j.job.running && j.job.error) {
				generating = false;
				genError = j.job.error;
				return;
			}
			await new Promise((r) => setTimeout(r, 800));
		}
		generating = false;
		genError = 'timed out waiting for mosaic';
	}

	async function startGenerate(force = false) {
		genError = null;
		generating = true;
		ready = false;
		stage = 'checking';
		const st = await layerStatus();
		if (!force && st.ready) {
			await loadMeta();
			return;
		}
		stage = 'start';
		const r = await fetch('/api/eval/make-full', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				pair_id: data.pairId,
				dataset,
				layers: [side],
				force
			})
		});
		if (!r.ok) {
			genError = await r.text();
			generating = false;
			return;
		}
		const body = await r.json();
		applyJob(body.state ?? body.job ?? null);
		if (body.cached || (body.state && !body.state.running && !body.state.error)) {
			await loadMeta();
			return;
		}
		await waitUntilReady();
	}

	onMount(() => {
		void startGenerate(false);
	});
</script>

{#if genError}
	<div class="empty err">
		Failed to build {side.toUpperCase()} mosaic: {genError}
		<button type="button" class="retry" onclick={() => startGenerate(true)}>Retry</button>
	</div>
{:else if !ready || !fullMeta}
	<div class="empty">
		Building unwarped {side.toUpperCase()} mosaic…
		<span class="prog">{done}/{total || 1} · {stage ?? '…'}</span>
	</div>
{:else}
	<NativeSlideViewer pairId={data.pairId} {side} {dataset} {fullMeta} />
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
