<script lang="ts">
	import PointCanvas from '$lib/PointCanvas.svelte';
	import { liveWholeUrl } from '$lib/liveCropUrl';
	import { getDeskew, applyDeskew, clearDeskew, type DeskewPoint } from '$lib/c2fClient';
	import { CNN_WIDTH, CNN_HEIGHT } from '$lib/types';

	interface Point {
		x: number;
		y: number;
	}

	let { data } = $props();
	const pairId = $derived(data.pairId);

	// Whole-image preview level: 2 => grid 4 => 2048x1376 (~4x the 512x344 base).
	const PREVIEW_LEVEL = 2;
	const PREVIEW_W = 2 ** PREVIEW_LEVEL * CNN_WIDTH;
	const PREVIEW_H = 2 ** PREVIEW_LEVEL * CNN_HEIGHT;
	const DISPLAY_W = 860;
	const DISPLAY_H = Math.round((DISPLAY_W * CNN_HEIGHT) / CNN_WIDTH);

	let hePoints = $state<Point[]>([]);
	let ihcPoints = $state<Point[]>([]);
	let busy = $state(false);
	let status = $state<{ msg: string; kind: 'ok' | 'err' } | null>(null);

	const heSrc = $derived(liveWholeUrl(pairId, 'he', PREVIEW_LEVEL));
	const ihcSrc = $derived(liveWholeUrl(pairId, 'ihc', PREVIEW_LEVEL));

	const pairs = $derived(Math.min(hePoints.length, ihcPoints.length));
	const balanced = $derived(hePoints.length === ihcPoints.length);
	const canApply = $derived(balanced && pairs >= 3 && !busy);

	$effect(() => {
		const pair = pairId;
		let stale = false;
		getDeskew(pair).then((d) => {
			if (stale) return;
			hePoints = d.points.map((p) => ({ x: p.he[0] * PREVIEW_W, y: p.he[1] * PREVIEW_H }));
			ihcPoints = d.points.map((p) => ({ x: p.ihc[0] * PREVIEW_W, y: p.ihc[1] * PREVIEW_H }));
		});
		return () => {
			stale = true;
		};
	});

	function addHe(x: number, y: number) {
		hePoints = [...hePoints, { x, y }];
	}
	function addIhc(x: number, y: number) {
		ihcPoints = [...ihcPoints, { x, y }];
	}

	function removeLast() {
		if (hePoints.length > ihcPoints.length) hePoints = hePoints.slice(0, -1);
		else if (ihcPoints.length > hePoints.length) ihcPoints = ihcPoints.slice(0, -1);
		else {
			hePoints = hePoints.slice(0, -1);
			ihcPoints = ihcPoints.slice(0, -1);
		}
	}

	function clearPoints() {
		hePoints = [];
		ihcPoints = [];
	}

	async function apply() {
		if (!canApply) return;
		busy = true;
		status = null;
		try {
			const points: DeskewPoint[] = [];
			for (let i = 0; i < pairs; i++) {
				points.push({
					he: [hePoints[i].x / PREVIEW_W, hePoints[i].y / PREVIEW_H],
					ihc: [ihcPoints[i].x / PREVIEW_W, ihcPoints[i].y / PREVIEW_H]
				});
			}
			const ok = await applyDeskew(pairId, PREVIEW_LEVEL, points);
			status = ok
				? { msg: `Deskew applied (${points.length} pairs). All IHC crops for this pair are now warped; FFT caches were cleared.`, kind: 'ok' }
				: { msg: 'Deskew failed.', kind: 'err' };
		} finally {
			busy = false;
		}
	}

	async function reset() {
		busy = true;
		status = null;
		try {
			await clearDeskew(pairId);
			clearPoints();
			status = { msg: 'Deskew cleared for this pair.', kind: 'ok' };
		} finally {
			busy = false;
		}
	}
</script>

<div class="deskew-page">
	<header class="dp-head">
		<div>
			<h1>Deskew · pair {pairId}</h1>
			<p class="dp-sub">
				Rotation-free global affine (translation + anisotropic stretch + shear). Click a landmark on
				HE, then its match on IHC. Place ≥3 pairs and Apply — it becomes an image warp of the moving
				(IHC) channel at every level, correcting strong skew that a per-tile shift cannot.
			</p>
		</div>
		<a class="dp-back" href={`/${pairId}/0`}>← Back to pair {pairId}</a>
	</header>

	<div class="dp-grid">
		<div class="dp-cell">
			<span class="dp-cap">HE (fixed) · {hePoints.length} points</span>
			<PointCanvas
				src={heSrc}
				active={true}
				points={hePoints}
				width={DISPLAY_W}
				height={DISPLAY_H}
				markerScale={0.1}
				onpoint={addHe}
			/>
		</div>
		<div class="dp-cell">
			<span class="dp-cap">IHC (moving) · {ihcPoints.length} points</span>
			<PointCanvas
				src={ihcSrc}
				active={true}
				points={ihcPoints}
				width={DISPLAY_W}
				height={DISPLAY_H}
				markerScale={0.1}
				onpoint={addIhc}
			/>
		</div>
	</div>

	<div class="dp-controls">
		<span class="dp-count" class:warn={!balanced}>
			{pairs} pair{pairs === 1 ? '' : 's'}{balanced ? '' : ' (place the matching point)'}
		</span>
		<button class="dp-btn dp-btn-primary" disabled={!canApply} onclick={apply}>
			{busy ? 'Applying…' : 'Apply deskew'}
		</button>
		<button
			class="dp-btn"
			disabled={busy || (hePoints.length === 0 && ihcPoints.length === 0)}
			onclick={removeLast}
		>Remove last</button>
		<button
			class="dp-btn"
			disabled={busy || (hePoints.length === 0 && ihcPoints.length === 0)}
			onclick={clearPoints}
		>Clear points</button>
		<button class="dp-btn dp-btn-ghost" disabled={busy} onclick={reset}>Reset deskew</button>
		{#if status}
			<span class="dp-status" class:err={status.kind === 'err'}>{status.msg}</span>
		{/if}
	</div>
</div>

<style>
	.deskew-page {
		height: 100%;
		overflow-y: auto;
		padding: 20px 24px;
		color: #e5e7eb;
		max-width: 1900px;
		margin: 0 auto;
	}
	.dp-head {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 24px;
		margin-bottom: 16px;
	}
	.dp-head h1 {
		font-size: 1.05rem;
		font-weight: 700;
		margin: 0 0 4px;
		color: #c4b5fd;
	}
	.dp-sub {
		font-size: 0.78rem;
		color: #9ca3af;
		max-width: 900px;
		margin: 0;
		line-height: 1.5;
	}
	.dp-back {
		flex: none;
		font-size: 0.8rem;
		color: #93c5fd;
		text-decoration: none;
		white-space: nowrap;
	}
	.dp-back:hover {
		text-decoration: underline;
	}
	.dp-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 24px;
		margin-bottom: 16px;
	}
	.dp-cell {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.dp-cap {
		font-size: 0.75rem;
		font-weight: 600;
		color: #9ca3af;
	}
	.dp-controls {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
	}
	.dp-count {
		font-size: 0.8rem;
		color: #9ca3af;
	}
	.dp-count.warn {
		color: #f59e0b;
	}
	.dp-btn {
		font-size: 0.8rem;
		padding: 7px 14px;
		border-radius: 6px;
		border: 1px solid #2a2d3a;
		background: #1b1e28;
		color: #d1d5db;
		cursor: pointer;
	}
	.dp-btn:disabled {
		opacity: 0.45;
		cursor: default;
	}
	.dp-btn-primary {
		border-color: #6366f1;
		background: #312e81;
		color: #e0e7ff;
	}
	.dp-btn-ghost {
		color: #9ca3af;
	}
	.dp-status {
		font-size: 0.8rem;
		color: #86efac;
	}
	.dp-status.err {
		color: #fca5a5;
	}
</style>
