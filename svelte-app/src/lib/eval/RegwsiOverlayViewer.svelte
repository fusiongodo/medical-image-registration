<script lang="ts">
	import { CNN_WIDTH, CNN_HEIGHT } from '$lib/types';

	export type MovingLayer = string;

	let {
		pairId,
		title,
		subtitle = '',
		movingLayer,
		fullMeta,
		movingReady = true
	}: {
		pairId: number;
		title: string;
		subtitle?: string;
		movingLayer: MovingLayer;
		fullMeta: { w: number; h: number; qw: number; qh: number; nq?: number };
		movingReady?: boolean;
	} = $props();

	const MAX_ZOOM = 40;
	const LEVEL = 5;
	const GRID = 2 ** LEVEL;

	let frame = $state<HTMLDivElement | null>(null);
	let frameW = $state(960);
	let frameH = $state(640);
	let imgW = $state(0);
	let imgH = $state(0);
	let qw = $state(0);
	let qh = $state(0);
	let nq = $state(2);
	let scale = $state(1);
	let baseScale = $state(1);
	let tx = $state(0);
	let ty = $state(0);
	let dragging = $state(false);
	let lastX = 0;
	let lastY = 0;

	let emphasis = $state<'he' | 'ihc' | null>(null);
	let showGrid = $state(true);
	let loaded = $state(0);

	const heOpacity = $derived(emphasis === 'ihc' ? 0 : 1);
	const ihcOpacity = $derived(emphasis === 'he' ? 0 : emphasis === 'ihc' ? 1 : 0.5);
	const quads = $derived(
		Array.from({ length: nq * nq }, (_, i) => ({ qy: Math.floor(i / nq), qx: i % nq }))
	);
	const totalImgs = $derived(nq * nq * (movingReady ? 2 : 1));
	const allLoaded = $derived(loaded >= totalImgs);

	const cellW = $derived(imgW > 0 ? imgW / GRID : CNN_WIDTH);
	const cellH = $derived(imgH > 0 ? imgH / GRID : CNN_HEIGHT);
	const vLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellW));
	const hLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellH));

	function quadSrc(layer: 'he' | MovingLayer, qy: number, qx: number) {
		return `/api/eval/full?pair=${pairId}&layer=${layer}&qy=${qy}&qx=${qx}`;
	}

	function fit() {
		if (!imgW || !imgH || !frameW || !frameH) return;
		baseScale = Math.min(frameW / imgW, frameH / imgH);
		scale = baseScale;
		tx = (frameW - imgW * baseScale) / 2;
		ty = (frameH - imgH * baseScale) / 2;
	}

	function onImgLoad() {
		loaded += 1;
	}

	function toggleEmphasis(side: 'he' | 'ihc') {
		emphasis = emphasis === side ? null : side;
	}

	$effect(() => {
		void pairId;
		void movingLayer;
		loaded = 0;
		imgW = fullMeta.w;
		imgH = fullMeta.h;
		qw = fullMeta.qw;
		qh = fullMeta.qh;
		nq = fullMeta.nq ?? 2;
		fit();
	});

	$effect(() => {
		if (!frame) return;
		const ro = new ResizeObserver((entries) => {
			const r = entries[0]?.contentRect;
			if (!r) return;
			frameW = r.width;
			frameH = r.height;
			fit();
		});
		ro.observe(frame);
		return () => ro.disconnect();
	});

	$effect(() => {
		function onKeyDown(e: KeyboardEvent) {
			if (!e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
			const target = e.target as HTMLElement | null;
			if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
			if (e.key === 'Q' || e.key === 'q') {
				e.preventDefault();
				toggleEmphasis('he');
			} else if (e.key === 'W' || e.key === 'w') {
				e.preventDefault();
				toggleEmphasis('ihc');
			}
		}
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});

	function onWheel(e: WheelEvent) {
		if (!frame || !imgW) return;
		e.preventDefault();
		const rect = frame.getBoundingClientRect();
		const cx = e.clientX - rect.left;
		const cy = e.clientY - rect.top;
		const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
		const next = Math.min(baseScale * MAX_ZOOM, Math.max(baseScale, scale * factor));
		tx = cx - ((cx - tx) * next) / scale;
		ty = cy - ((cy - ty) * next) / scale;
		scale = next;
		if (scale === baseScale) fit();
	}

	function onPointerDown(e: PointerEvent) {
		dragging = true;
		lastX = e.clientX;
		lastY = e.clientY;
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
	}
	function onPointerMove(e: PointerEvent) {
		if (!dragging) return;
		tx += e.clientX - lastX;
		ty += e.clientY - lastY;
		lastX = e.clientX;
		lastY = e.clientY;
	}
	function onPointerUp(e: PointerEvent) {
		dragging = false;
		(e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
	}
</script>

<div class="page">
	<header class="head">
		<div>
			<h1>{title}</h1>
			{#if subtitle}
				<p class="sub">{subtitle}</p>
			{/if}
			<p class="hint">Shift+Q solo HE · Shift+W solo IHC · scroll zoom · drag pan</p>
		</div>
		<div class="controls">
			<span class="emph-pills">
				<button
					type="button"
					class="emph-pill"
					class:active={emphasis === 'he'}
					onclick={() => toggleEmphasis('he')}
					title="Show only HE (Shift+Q)"
				>HE</button>
				<button
					type="button"
					class="emph-pill"
					class:active={emphasis === 'ihc'}
					onclick={() => toggleEmphasis('ihc')}
					title="Show only IHC (Shift+W)"
				>IHC</button>
			</span>
			<label class="chk">
				<input type="checkbox" bind:checked={showGrid} />
				L5 grid
			</label>
			<button type="button" class="reset" onclick={fit} disabled={!allLoaded}>Reset view</button>
			<span class="zoom">{allLoaded ? `${(scale / baseScale).toFixed(1)}×` : `${loaded}/${totalImgs}`}</span>
		</div>
	</header>

	{#if !movingReady}
		<div class="empty">Moving layer not ready yet.</div>
	{:else}
		<div
			class="frame"
			bind:this={frame}
			onwheel={onWheel}
			onpointerdown={onPointerDown}
			onpointermove={onPointerMove}
			onpointerup={onPointerUp}
			role="presentation"
		>
			<div
				class="stack"
				style:transform={`translate(${tx}px, ${ty}px) scale(${scale})`}
				style:width={`${imgW}px`}
				style:height={`${imgH}px`}
			>
				{#key `${pairId}-${nq}-${movingLayer}`}
					{#each quads as q (`${q.qy}_${q.qx}`)}
						{@const left = q.qx === 0 ? 0 : q.qx * qw}
						{@const top = q.qy === 0 ? 0 : q.qy * qh}
						{@const cellWw = q.qx < nq - 1 ? qw : imgW - q.qx * qw}
						{@const cellHh = q.qy < nq - 1 ? qh : imgH - q.qy * qh}
						<div
							class="cell"
							style:left={`${left}px`}
							style:top={`${top}px`}
							style:width={`${cellWw}px`}
							style:height={`${cellHh}px`}
						>
							<img
								class="layer he"
								src={quadSrc('he', q.qy, q.qx)}
								alt=""
								draggable="false"
								onload={onImgLoad}
								style:opacity={heOpacity}
							/>
							<img
								class="layer ihc"
								src={quadSrc(movingLayer, q.qy, q.qx)}
								alt=""
								draggable="false"
								onload={onImgLoad}
								style:opacity={ihcOpacity}
							/>
						</div>
					{/each}
				{/key}
				{#if showGrid && imgW && imgH}
					<svg class="grid" width={imgW} height={imgH} viewBox={`0 0 ${imgW} ${imgH}`} aria-hidden="true">
						{#each vLines as x}
							<line x1={x} y1={0} x2={x} y2={imgH} />
						{/each}
						{#each hLines as y}
							<line x1={0} y1={y} x2={imgW} y2={y} />
						{/each}
					</svg>
				{/if}
			</div>
			{#if !allLoaded}
				<span class="loading">loading {loaded}/{totalImgs}…</span>
			{/if}
		</div>
	{/if}
</div>

<style>
	.page {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
		padding: 0.65rem 0.85rem 0.85rem;
		box-sizing: border-box;
		gap: 0.5rem;
		background: #0f1117;
		color: #e8eaf0;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 1rem;
		flex-shrink: 0;
	}
	h1 {
		margin: 0 0 0.15rem;
		font-size: 1.1rem;
		font-weight: 600;
	}
	.sub {
		margin: 0;
		color: #9ca3af;
		font-size: 0.8rem;
		max-width: 48rem;
		line-height: 1.35;
	}
	.hint {
		margin: 0.2rem 0 0;
		color: #6b7280;
		font-size: 0.72rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.65rem 0.85rem;
		font-size: 0.85rem;
	}
	.emph-pills {
		display: inline-flex;
		gap: 0.3rem;
	}
	.emph-pill {
		padding: 0.2rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 3px;
		background: #181b23;
		cursor: pointer;
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		color: #9ca3af;
	}
	.emph-pill.active {
		border-color: #5b8def;
		background: #1e2740;
		color: #e8eaf0;
	}
	.chk {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		cursor: pointer;
		color: #9ca3af;
		font-size: 0.8rem;
	}
	.reset {
		padding: 0.3rem 0.65rem;
		border: 1px solid #2a2d3a;
		border-radius: 3px;
		background: #181b23;
		cursor: pointer;
		font-size: 0.8rem;
		color: #c4c9d4;
	}
	.reset:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.zoom {
		font-variant-numeric: tabular-nums;
		color: #6b7280;
		min-width: 2.5rem;
	}
	.empty {
		padding: 1.5rem;
		border: 1px dashed #2a2d3a;
		border-radius: 4px;
		color: #9ca3af;
	}
	.frame {
		flex: 1;
		position: relative;
		min-height: 0;
		border: 1px solid #2a2d3a;
		background: #1a1a1a;
		overflow: hidden;
		cursor: grab;
		touch-action: none;
		user-select: none;
	}
	.frame:active {
		cursor: grabbing;
	}
	.stack {
		position: absolute;
		top: 0;
		left: 0;
		transform-origin: 0 0;
		will-change: transform;
	}
	.cell {
		position: absolute;
		overflow: hidden;
	}
	.layer {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		display: block;
		transition: opacity 0.08s linear;
	}
	.grid {
		position: absolute;
		top: 0;
		left: 0;
		pointer-events: none;
		overflow: visible;
	}
	.grid line {
		stroke: rgba(255, 255, 255, 0.4);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.loading {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		color: #bbb;
		font-size: 0.85rem;
		pointer-events: none;
	}
</style>
