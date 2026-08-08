<script lang="ts">
	import { CNN_WIDTH, CNN_HEIGHT } from '$lib/types';
	import type { AnnotateSide, Landmark } from '$lib/eval/annotateSession';

	type FullMeta = { w: number; h: number; qw: number; qh: number; nq?: number };

	let {
		pairId,
		side,
		meta,
		landmarks,
		pendingHe,
		active,
		onAnnotate,
		dataset = 'muromi'
	}: {
		pairId: number;
		side: AnnotateSide;
		meta: FullMeta;
		landmarks: Landmark[];
		pendingHe: [number, number] | null;
		active: boolean;
		onAnnotate: (pt: [number, number]) => void;
		dataset?: 'muromi' | 'acrobat';
	} = $props();

	const MAX_ZOOM = 40;
	const LEVEL = 5;
	const GRID = 2 ** LEVEL;

	let frame = $state<HTMLDivElement | null>(null);
	let frameW = $state(960);
	let frameH = $state(640);
	let scale = $state(1);
	let baseScale = $state(1);
	let tx = $state(0);
	let ty = $state(0);
	let dragging = $state(false);
	let lastX = 0;
	let lastY = 0;
	let downX = 0;
	let downY = 0;
	let showGrid = $state(true);
	let loaded = $state(0);

	const imgW = $derived(meta.w);
	const imgH = $derived(meta.h);
	const qw = $derived(meta.qw);
	const qh = $derived(meta.qh);
	const nq = $derived(meta.nq ?? 2);
	const quads = $derived(
		Array.from({ length: nq * nq }, (_, i) => ({ qy: Math.floor(i / nq), qx: i % nq }))
	);
	const totalImgs = $derived(nq * nq);
	const allLoaded = $derived(loaded >= totalImgs);
	const cellW = $derived(imgW > 0 ? imgW / GRID : CNN_WIDTH);
	const cellH = $derived(imgH > 0 ? imgH / GRID : CNN_HEIGHT);
	const vLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellW));
	const hLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellH));

	function quadSrc(qy: number, qx: number) {
		return `/api/eval/full?pair=${pairId}&layer=${side}&qy=${qy}&qx=${qx}&dataset=${dataset}`;
	}

	function fit() {
		if (!imgW || !imgH || !frameW || !frameH) return;
		baseScale = Math.min(frameW / imgW, frameH / imgH);
		scale = baseScale;
		tx = (frameW - imgW * baseScale) / 2;
		ty = (frameH - imgH * baseScale) / 2;
	}

	function clientToNorm(e: PointerEvent): [number, number] | null {
		if (!frame || !imgW || !imgH || scale === 0) return null;
		const rect = frame.getBoundingClientRect();
		const cx = e.clientX - rect.left;
		const cy = e.clientY - rect.top;
		const ix = (cx - tx) / scale;
		const iy = (cy - ty) / scale;
		if (ix < 0 || iy < 0 || ix > imgW || iy > imgH) return null;
		return [ix / imgW, iy / imgH];
	}

	function onFrameClick(e: PointerEvent) {
		if (dragging || !active) return;
		const pt = clientToNorm(e);
		if (!pt) return;
		onAnnotate(pt);
	}

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
		downX = e.clientX;
		downY = e.clientY;
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
		const moved = Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY);
		dragging = false;
		(e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
		if (moved < 5) onFrameClick(e);
	}

	$effect(() => {
		void pairId;
		void side;
		loaded = 0;
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
</script>

<div class="wrap">
	<div class="controls">
		<label class="chk">
			<input type="checkbox" bind:checked={showGrid} />
			L5 grid
		</label>
		<button class="btn" onclick={fit} disabled={!allLoaded}>Reset view</button>
		<span class="zoom">{allLoaded ? `${(scale / baseScale).toFixed(1)}×` : `${loaded}/${totalImgs}`}</span>
	</div>
	<div
		class="frame"
		class:active
		class:waiting={!active}
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
			{#key `${pairId}-${side}-${nq}`}
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
							class="layer"
							src={quadSrc(q.qy, q.qx)}
							alt=""
							draggable="false"
							onload={() => (loaded += 1)}
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
			<svg class="marks" width={imgW} height={imgH} viewBox={`0 0 ${imgW} ${imgH}`} aria-hidden="true">
				{#each landmarks as lm, i}
					{@const pt = side === 'he' ? lm.he : lm.ihc}
					{@const x = pt[0] * imgW}
					{@const y = pt[1] * imgH}
					<circle class="mk fill" class:he={side === 'he'} class:ihc={side === 'ihc'} cx={x} cy={y} r={5 / scale} />
					<circle class="mk" class:he={side === 'he'} class:ihc={side === 'ihc'} cx={x} cy={y} r={10 / scale} />
					<text
						class="mk-label"
						class:he={side === 'he'}
						class:ihc={side === 'ihc'}
						x={x + 12 / scale}
						y={y - 12 / scale}
						style:font-size={`${14 / scale}px`}>{i + 1}</text
					>
				{/each}
				{#if side === 'he' && pendingHe}
					{@const px = pendingHe[0] * imgW}
					{@const py = pendingHe[1] * imgH}
					{@const arm = 18 / scale}
					<line class="pending-cross" x1={px - arm} y1={py} x2={px + arm} y2={py} />
					<line class="pending-cross" x1={px} y1={py - arm} x2={px} y2={py + arm} />
					<circle class="mk fill pending" cx={px} cy={py} r={6 / scale} />
					<circle class="mk pending" cx={px} cy={py} r={14 / scale} />
					<text
						class="mk-label he"
						x={px + 16 / scale}
						y={py - 16 / scale}
						style:font-size={`${14 / scale}px`}>●</text
					>
				{/if}
			</svg>
		</div>
		{#if !allLoaded}
			<span class="loading">loading {loaded}/{totalImgs}…</span>
		{/if}
	</div>
</div>

<style>
	.wrap {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		flex: 1;
		min-height: 0;
	}
	.controls {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		font-size: 0.85rem;
	}
	.chk {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		cursor: pointer;
		color: #9ca3af;
		font-size: 0.8rem;
	}
	.btn {
		margin-left: auto;
		padding: 0.3rem 0.65rem;
		border: 1px solid #2a2d3a;
		border-radius: 3px;
		background: #181b23;
		color: #c4c9d4;
		cursor: pointer;
		font-size: 0.8rem;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.zoom {
		font-variant-numeric: tabular-nums;
		color: #6b7280;
		min-width: 2.5rem;
	}
	.frame {
		position: relative;
		flex: 1;
		min-height: 320px;
		border: 1px solid #2a2d3a;
		background: #0a0b0f;
		overflow: hidden;
		touch-action: none;
		user-select: none;
	}
	.frame.active {
		cursor: crosshair;
		border-color: #22c55e;
		box-shadow: inset 0 0 0 2px rgba(34, 197, 94, 0.45);
	}
	.frame.waiting {
		cursor: not-allowed;
		border-color: #f59e0b;
		box-shadow: inset 0 0 0 2px rgba(245, 158, 11, 0.35);
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
	}
	.grid,
	.marks {
		position: absolute;
		top: 0;
		left: 0;
		pointer-events: none;
		overflow: visible;
	}
	.grid line {
		stroke: rgba(255, 255, 255, 0.35);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.mk {
		fill: none;
		stroke-width: 2.5;
		vector-effect: non-scaling-stroke;
	}
	.mk.fill {
		stroke: none;
	}
	.mk.he {
		stroke: #2563eb;
	}
	.mk.fill.he {
		fill: #3b82f6;
	}
	.mk.ihc {
		stroke: #d97706;
	}
	.mk.fill.ihc {
		fill: #f59e0b;
	}
	.mk.pending {
		stroke: #ef4444;
		stroke-dasharray: 5 4;
	}
	.mk.fill.pending {
		fill: #ef4444;
		stroke: none;
	}
	.pending-cross {
		stroke: #ef4444;
		stroke-width: 2.5;
		vector-effect: non-scaling-stroke;
	}
	.mk-label {
		font-weight: 700;
		paint-order: stroke fill;
		stroke: #0a0b0f;
		stroke-width: 3px;
	}
	.mk-label.he {
		fill: #93c5fd;
	}
	.mk-label.ihc {
		fill: #fcd34d;
	}
	.loading {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		pointer-events: none;
		font-size: 0.9rem;
		font-weight: 600;
		color: #9ca3af;
	}
</style>
