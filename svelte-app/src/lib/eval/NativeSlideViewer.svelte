<script lang="ts">
	import { CNN_WIDTH, CNN_HEIGHT } from '$lib/types';

	type Side = 'he' | 'ihc';

	let {
		pairId,
		side,
		fullMeta,
		dataset = 'muromi'
	}: {
		pairId: number;
		side: Side;
		fullMeta: { w: number; h: number; qw: number; qh: number; nq?: number };
		dataset?: 'muromi' | 'acrobat' | 'anhir';
	} = $props();

	const MAX_ZOOM = 40;
	const LEVEL = 5;
	const GRID = 2 ** LEVEL;
	const HE_LM = '#7c3aed';
	const IHC_LM = '#f97316';

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
	let showGrid = $state(true);
	let showLandmarks = $state(true);
	let loaded = $state(0);
	let landmarks = $state<{ he: [number, number]; ihc?: [number, number] }[]>([]);

	const quads = $derived(
		Array.from({ length: nq * nq }, (_, i) => ({ qy: Math.floor(i / nq), qx: i % nq }))
	);
	const totalImgs = $derived(nq * nq);
	const allLoaded = $derived(loaded >= totalImgs);
	const cellW = $derived(imgW > 0 ? imgW / GRID : CNN_WIDTH);
	const cellH = $derived(imgH > 0 ? imgH / GRID : CNN_HEIGHT);
	const vLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellW));
	const hLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellH));
	const lmR = $derived(6 / Math.max(scale, 1e-6));
	const lmFont = $derived(11 / Math.max(scale, 1e-6));
	const showLmLabels = $derived(landmarks.length > 0 && landmarks.length <= 40);
	const fill = $derived(side === 'he' ? HE_LM : IHC_LM);

	function inUnit(pt: [number, number] | undefined): pt is [number, number] {
		return (
			!!pt &&
			pt.length === 2 &&
			Number.isFinite(pt[0]) &&
			Number.isFinite(pt[1]) &&
			pt[0] >= 0 &&
			pt[0] <= 1 &&
			pt[1] >= 0 &&
			pt[1] <= 1
		);
	}

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

	$effect(() => {
		void pairId;
		void side;
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
		const p = pairId;
		const ds = dataset;
		let cancelled = false;
		landmarks = [];
		void fetch(`/api/eval/landmarks?pair=${p}&dataset=${ds}`)
			.then((r) => (r.ok ? r.json() : null))
			.then((d) => {
				if (cancelled) return;
				landmarks = Array.isArray(d?.points) ? d.points : [];
			})
			.catch(() => {
				if (!cancelled) landmarks = [];
			});
		return () => {
			cancelled = true;
		};
	});
</script>

<div class="page">
	<header class="head">
		<div>
			<h1>Native {side.toUpperCase()} · pair {pairId}</h1>
			<p class="sub">Unwarped {side.toUpperCase()} canvas. Landmarks in original image coords.</p>
			<p class="hint">scroll zoom · drag pan</p>
		</div>
		<div class="controls">
			<a class="nav" href={`/eval/${pairId}/native/${side === 'he' ? 'ihc' : 'he'}?dataset=${dataset}`}>
				Open {side === 'he' ? 'IHC' : 'HE'}
			</a>
			<label class="chk">
				<input type="checkbox" bind:checked={showGrid} />
				L5 grid
			</label>
			<label class="chk">
				<input type="checkbox" bind:checked={showLandmarks} />
				Landmarks
			</label>
			<button type="button" class="reset" onclick={fit} disabled={!allLoaded}>Reset view</button>
			<span class="zoom">{allLoaded ? `${(scale / baseScale).toFixed(1)}×` : `${loaded}/${totalImgs}`}</span>
		</div>
	</header>

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
			{#key `${pairId}-${nq}-${side}`}
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
						<line x1={0} y1={y} x2={imgW} y2={imgH} />
					{/each}
				</svg>
			{/if}
			{#if showLandmarks && imgW && imgH && landmarks.length}
				<svg class="lms" width={imgW} height={imgH} viewBox={`0 0 ${imgW} ${imgH}`} aria-hidden="true">
					{#each landmarks as p, i}
						{@const pt = side === 'he' ? p.he : p.ihc}
						{#if inUnit(pt)}
							{@const x = pt[0] * imgW}
							{@const y = pt[1] * imgH}
							<circle cx={x} cy={y} r={lmR} fill={fill} />
							{#if showLmLabels}
								<text x={x + lmR * 1.4} y={y} font-size={lmFont}>{i}</text>
							{/if}
						{/if}
					{/each}
				</svg>
			{/if}
		</div>
		{#if !allLoaded}
			<span class="loading">loading {loaded}/{totalImgs}…</span>
		{/if}
	</div>
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
	.chk {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		color: #9ca3af;
		font-size: 0.75rem;
		cursor: pointer;
	}
	.nav {
		color: #93c5fd;
		font-size: 0.75rem;
		text-decoration: none;
	}
	.reset {
		padding: 0.2rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 3px;
		background: #181b23;
		color: #e8eaf0;
		cursor: pointer;
		font-size: 0.75rem;
	}
	.reset:disabled {
		opacity: 0.45;
		cursor: default;
	}
	.zoom {
		color: #6b7280;
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
	}
	.frame {
		position: relative;
		flex: 1;
		min-height: 0;
		overflow: hidden;
		background: #0a0b10;
		border: 1px solid #1f2330;
		cursor: grab;
		touch-action: none;
	}
	.frame:active {
		cursor: grabbing;
	}
	.stack {
		position: absolute;
		top: 0;
		left: 0;
		transform-origin: 0 0;
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
	.lms {
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
	.lms circle {
		stroke: #0f1117;
		stroke-width: 1.25;
		vector-effect: non-scaling-stroke;
	}
	.lms text {
		fill: #e8eaf0;
		paint-order: stroke;
		stroke: #0f1117;
		stroke-width: 3;
		vector-effect: non-scaling-stroke;
		dominant-baseline: middle;
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
