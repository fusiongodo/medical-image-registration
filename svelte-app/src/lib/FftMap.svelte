<script lang="ts">
	let { src }: { src: string } = $props();

	const MODAL_W = 760;
	const MODAL_H = Math.round((MODAL_W * 344) / 512);
	const MAX_ZOOM = 40;

	interface Peak {
		dx: number;
		dy: number;
		psr: number;
		px: number;
		py: number;
	}
	interface MapData {
		image: string;
		w: number;
		h: number;
		cx: number;
		cy: number;
		peaks: Peak[];
		chosen: { dx: number; dy: number; px: number; py: number } | null;
	}

	let data = $state<MapData | null>(null);
	let loadErr = $state<string | null>(null);
	let open = $state(false);
	let showMarkers = $state(true);

	let frame = $state<HTMLDivElement | null>(null);
	let scale = $state(1);
	let tx = $state(0);
	let ty = $state(0);
	let baseScale = $state(1);
	let dragging = $state(false);
	let lastX = 0;
	let lastY = 0;

	function fit(d: MapData) {
		baseScale = Math.min(MODAL_W / d.w, MODAL_H / d.h);
		scale = baseScale;
		tx = (MODAL_W - d.w * baseScale) / 2;
		ty = (MODAL_H - d.h * baseScale) / 2;
	}

	$effect(() => {
		const url = src;
		data = null;
		loadErr = null;
		open = false;
		if (!url) return;
		let cancelled = false;
		fetch(url)
			.then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
			.then((d: MapData) => {
				if (cancelled) return;
				data = d;
			})
			.catch((e) => {
				if (cancelled) return;
				loadErr = e instanceof Error ? e.message : 'load failed';
			});
		return () => {
			cancelled = true;
		};
	});

	function openWidget() {
		if (!data) return;
		fit(data);
		open = true;
	}

	$effect(() => {
		if (!open) return;
		function onKey(e: KeyboardEvent) {
			if (e.key === 'Escape') open = false;
		}
		window.addEventListener('keydown', onKey);
		return () => window.removeEventListener('keydown', onKey);
	});

	function onWheel(e: WheelEvent) {
		if (!frame) return;
		e.preventDefault();
		const rect = frame.getBoundingClientRect();
		const cx = e.clientX - rect.left;
		const cy = e.clientY - rect.top;
		const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
		const next = Math.min(baseScale * MAX_ZOOM, Math.max(baseScale, scale * factor));
		tx = cx - ((cx - tx) * next) / scale;
		ty = cy - ((cy - ty) * next) / scale;
		scale = next;
		if (data && scale === baseScale) fit(data);
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
	function reset() {
		if (data) fit(data);
	}

	const sx = (px: number) => tx + px * scale;
	const sy = (py: number) => ty + py * scale;
</script>

{#if src}
	<button class="fft-thumb" onclick={openWidget} title="Open FFT surface explorer" disabled={!data}>
		{#if data}
			<img class="thumb-img" src={data.image} alt="FFT surface thumbnail" draggable="false" />
			<span class="thumb-hint">⤢</span>
		{:else if loadErr}
			<span class="thumb-msg err">FFT ✕</span>
		{:else}
			<span class="thumb-msg">…</span>
		{/if}
	</button>
{/if}

{#if open && data}
	<div
		class="modal-backdrop"
		onclick={(e) => {
			if (e.target === e.currentTarget) open = false;
		}}
		role="presentation"
	>
		<div class="modal" role="dialog" aria-modal="true" tabindex="-1">
			<div class="modal-head">
				<span class="modal-title">FFT phase-correlation surface</span>
				<span class="modal-sub">scroll = zoom · drag = pan · double-click = reset</span>
				<button
					class="head-btn"
					class:off={!showMarkers}
					onclick={() => (showMarkers = !showMarkers)}
				>{showMarkers ? '◉ markers' : '○ markers'}</button>
				<button class="head-btn" onclick={() => (open = false)}>✕</button>
			</div>
			<div
				class="fft-frame"
				class:dragging
				bind:this={frame}
				style:width={`${MODAL_W}px`}
				style:height={`${MODAL_H}px`}
				onwheel={onWheel}
				onpointerdown={onPointerDown}
				onpointermove={onPointerMove}
				onpointerup={onPointerUp}
				onpointerleave={onPointerUp}
				ondblclick={reset}
				role="presentation"
			>
				<img
					class="fft-img"
					src={data.image}
					alt="FFT phase-correlation surface"
					draggable="false"
					style:width={`${data.w}px`}
					style:height={`${data.h}px`}
					style:transform={`translate(${tx}px, ${ty}px) scale(${scale})`}
				/>
				{#if showMarkers}
					<svg class="markers" width={MODAL_W} height={MODAL_H}>
						<g class="zero">
							<line x1={sx(data.cx) - 6} y1={sy(data.cy)} x2={sx(data.cx) + 6} y2={sy(data.cy)} />
							<line x1={sx(data.cx)} y1={sy(data.cy) - 6} x2={sx(data.cx)} y2={sy(data.cy) + 6} />
						</g>
						{#each data.peaks as pk, rank}
							<g class="peak" class:top={rank === 0}>
								<circle cx={sx(pk.px)} cy={sy(pk.py)} r="7" />
								<text x={sx(pk.px) + 9} y={sy(pk.py) - 7}>{rank}</text>
							</g>
						{/each}
						{#if data.chosen}
							<circle class="chosen" cx={sx(data.chosen.px)} cy={sy(data.chosen.py)} r="10" />
						{/if}
					</svg>
				{/if}
				<span class="fft-zoom-badge">{(scale / baseScale).toFixed(1)}×</span>
			</div>
			<div class="legend">
				<span class="legend-item"><span class="swatch grey"></span>zero shift</span>
				<span class="legend-item"><span class="swatch red"></span>top peak (rank 0)</span>
				<span class="legend-item"><span class="swatch blue"></span>chosen (current pick)</span>
			</div>
		</div>
	</div>
{/if}

<style>
	.fft-thumb {
		all: unset;
		position: relative;
		width: 132px;
		height: 88px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
		overflow: hidden;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
	}
	.fft-thumb:hover {
		border-color: #6366f1;
	}
	.fft-thumb:disabled {
		cursor: default;
		opacity: 0.6;
	}

	.thumb-img {
		width: 100%;
		height: 100%;
		object-fit: contain;
		display: block;
		image-rendering: pixelated;
	}

	.thumb-hint {
		position: absolute;
		bottom: 2px;
		right: 4px;
		font-size: 0.7rem;
		color: #e8eaf0;
		text-shadow: 0 0 3px #000;
		pointer-events: none;
	}

	.thumb-msg {
		font-size: 0.7rem;
		color: #6b7280;
	}
	.thumb-msg.err {
		color: #ef4444;
	}

	.modal-backdrop {
		position: fixed;
		inset: 0;
		z-index: 60;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.6);
	}

	.modal {
		background: #131520;
		border: 1px solid #2a2d3a;
		border-radius: 10px;
		padding: 12px;
		box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55);
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.modal-head {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.modal-title {
		font-size: 0.8rem;
		font-weight: 700;
		color: #e8eaf0;
	}
	.modal-sub {
		font-size: 0.65rem;
		color: #6b7280;
		margin-right: auto;
	}

	.head-btn {
		all: unset;
		cursor: pointer;
		padding: 3px 9px;
		border-radius: 5px;
		border: 1px solid #4b5563;
		background: #1e2130;
		color: #cbd5e1;
		font-size: 0.68rem;
	}
	.head-btn:hover {
		border-color: #6366f1;
	}
	.head-btn.off {
		color: #6b7280;
	}

	.fft-frame {
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
		overflow: hidden;
		position: relative;
		cursor: grab;
		touch-action: none;
	}
	.fft-frame.dragging {
		cursor: grabbing;
	}

	.fft-img {
		position: absolute;
		top: 0;
		left: 0;
		display: block;
		transform-origin: 0 0;
		image-rendering: pixelated;
		user-select: none;
		-webkit-user-drag: none;
	}

	.markers {
		position: absolute;
		inset: 0;
		pointer-events: none;
	}
	.markers .zero line {
		stroke: #969696;
		stroke-width: 1.5;
	}
	.markers .peak circle {
		fill: none;
		stroke: #e6e6e6;
		stroke-width: 1.5;
	}
	.markers .peak.top circle {
		stroke: #ff2d2d;
		stroke-width: 2;
	}
	.markers .peak text {
		fill: #e6e6e6;
		font-size: 11px;
		font-family: system-ui, sans-serif;
	}
	.markers .chosen {
		fill: none;
		stroke: #2d8cff;
		stroke-width: 2;
	}

	.fft-zoom-badge {
		position: absolute;
		top: 4px;
		right: 6px;
		padding: 1px 5px;
		border-radius: 4px;
		background: rgba(15, 17, 23, 0.7);
		color: #9ca3af;
		font-size: 0.62rem;
		font-variant-numeric: tabular-nums;
		pointer-events: none;
	}

	.legend {
		display: flex;
		gap: 14px;
		font-size: 0.65rem;
		color: #9ca3af;
	}
	.legend-item {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.swatch {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		flex-shrink: 0;
	}
	.swatch.grey {
		background: #969696;
	}
	.swatch.red {
		background: #ff0000;
	}
	.swatch.blue {
		background: #0078ff;
	}
</style>
