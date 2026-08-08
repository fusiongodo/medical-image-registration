<script lang="ts">
	import { CNN_WIDTH, CNN_HEIGHT } from '$lib/types';

	let { data } = $props();
	const pairId = $derived(data.pairId);

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
	let showWarped = $state(false);
	let loaded = $state(0);

	const warpedReady = $derived(!!data.warpedReady);
	const heOpacity = $derived(emphasis === 'ihc' ? 0 : 1);
	const ihcOpacity = $derived(emphasis === 'he' ? 0 : emphasis === 'ihc' ? 1 : 0.5);
	const quads = $derived(
		Array.from({ length: nq * nq }, (_, i) => ({ qy: Math.floor(i / nq), qx: i % nq }))
	);
	const useWarped = $derived(showWarped && warpedReady);
	const totalImgs = $derived(nq * nq * (useWarped ? 3 : 2));
	const allLoaded = $derived(loaded >= totalImgs);

	const cellW = $derived(imgW > 0 ? imgW / GRID : CNN_WIDTH);
	const cellH = $derived(imgH > 0 ? imgH / GRID : CNN_HEIGHT);
	const vLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellW));
	const hLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellH));

	function quadSrc(layer: 'he' | 'ihc' | 'ihc_warped', qy: number, qx: number) {
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
		loaded = 0;
		const meta = data.fullMeta;
		if (meta) {
			imgW = meta.w;
			imgH = meta.h;
			qw = meta.qw;
			qh = meta.qh;
			nq = meta.nq ?? 2;
			fit();
		} else {
			imgW = 0;
			imgH = 0;
			qw = 0;
			qh = 0;
			nq = 2;
		}
	});

	$effect(() => {
		void useWarped;
		loaded = 0;
	});

	$effect(() => {
		if (!warpedReady && showWarped) showWarped = false;
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
			<h1>regWSI · pair {pairId}</h1>
			<p class="sub">
				HE / raw IHC overlay. Warped IHC when available. Shift+Q/W solo HE / IHC. Use Annotate for
				TRE landmarks.
			</p>
		</div>
		<nav class="nav">
			<a href={`/eval/${pairId}/annotate`}>
				Annotate landmarks{#if data.landmarkCount > 0}
					<span class="lm-count"> · {data.landmarkCount}</span>
				{/if}
			</a>
			<a href="/eval">All pairs</a>
		</nav>
	</header>

	{#if !data.fullReady || !data.fullMeta}
		<div class="empty">
			No HE + IHC mosaic for this pair yet. Run:
			<code>python regWSI/make_full.py {pairId} --layers he ihc</code>
			{#if data.ready}
				<span class="hint">(registration exists; mosaic only)</span>
			{/if}
		</div>
	{:else}
		<div class="controls">
			<span class="emph-pills">
				<button
					class="emph-pill"
					class:active={emphasis === 'ihc'}
					onclick={() => toggleEmphasis('he')}
					title="Show only HE (Shift+Q); press again for overlay"
				>HE</button>
				<button
					class="emph-pill"
					class:active={emphasis === 'he'}
					onclick={() => toggleEmphasis('ihc')}
					title="Show only IHC (Shift+W); press again for overlay"
				>IHC</button>
			</span>
			<label class="chk">
				<input type="checkbox" bind:checked={showGrid} />
				L5 grid
			</label>
			<label class="chk" class:disabled={!warpedReady} title={warpedReady ? 'Toggle warped IHC' : 'No warped IHC mosaic (lean sync)'}>
				<input type="checkbox" bind:checked={showWarped} disabled={!warpedReady} />
				warped IHC
			</label>
			<button class="reset" onclick={fit} disabled={!allLoaded}>Reset view</button>
			<span class="zoom">{allLoaded ? `${(scale / baseScale).toFixed(1)}×` : `${loaded}/${totalImgs}`}</span>
		</div>

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
				{#key `${pairId}-${nq}-${useWarped}`}
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
								src={quadSrc('ihc', q.qy, q.qx)}
								alt=""
								draggable="false"
								onload={onImgLoad}
								style:opacity={useWarped ? 0 : ihcOpacity}
								style:visibility={useWarped ? 'hidden' : 'visible'}
							/>
							{#if useWarped}
								<img
									class="layer ihc"
									src={quadSrc('ihc_warped', q.qy, q.qx)}
									alt=""
									draggable="false"
									onload={onImgLoad}
									style:opacity={ihcOpacity}
								/>
							{/if}
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
		padding: 1rem 1.25rem 1.25rem;
		box-sizing: border-box;
		gap: 0.75rem;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 1rem;
	}
	h1 {
		margin: 0 0 0.25rem;
		font-size: 1.25rem;
		font-weight: 600;
	}
	.sub {
		margin: 0;
		color: #666;
		font-size: 0.85rem;
		max-width: 44rem;
		line-height: 1.4;
	}
	.nav {
		display: flex;
		gap: 0.75rem;
		font-size: 0.85rem;
		white-space: nowrap;
	}
	.nav a {
		color: #444;
	}
	.empty {
		padding: 1.5rem;
		border: 1px dashed #ccc;
		border-radius: 4px;
		color: #555;
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.empty code {
		display: block;
		margin-top: 0.5rem;
		padding: 0.5rem 0.65rem;
		background: #f4f4f4;
		border-radius: 3px;
		font-size: 0.8rem;
	}
	.empty .hint {
		display: block;
		margin-top: 0.5rem;
		color: #888;
		font-size: 0.8rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.75rem 1rem;
		font-size: 0.85rem;
	}
	.emph-pills {
		display: inline-flex;
		gap: 0.3rem;
	}
	.emph-pill {
		padding: 0.2rem 0.55rem;
		border: 1px solid #ccc;
		border-radius: 3px;
		background: #fff;
		cursor: pointer;
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		color: #555;
	}
	.emph-pill.active {
		border-color: #444;
		background: #222;
		color: #fff;
	}
	.chk {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		cursor: pointer;
		color: #555;
		font-size: 0.8rem;
	}
	.chk.disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.reset {
		margin-left: auto;
		padding: 0.3rem 0.65rem;
		border: 1px solid #ccc;
		border-radius: 3px;
		background: #fff;
		cursor: pointer;
		font-size: 0.8rem;
	}
	.reset:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.zoom {
		font-variant-numeric: tabular-nums;
		color: #777;
		min-width: 2.5rem;
	}
	.frame {
		flex: 1;
		position: relative;
		min-height: 320px;
		border: 1px solid #ddd;
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
	}
	.layer.he,
	.layer.ihc {
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
