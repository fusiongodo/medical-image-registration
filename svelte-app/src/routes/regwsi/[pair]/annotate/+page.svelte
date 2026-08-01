<script lang="ts">
	import { CNN_WIDTH, CNN_HEIGHT } from '$lib/types';

	let { data } = $props();
	const pairId = $derived(data.pairId);

	const MAX_ZOOM = 40;
	const LEVEL = 5;
	const GRID = 2 ** LEVEL;

	interface Landmark {
		he: [number, number];
		ihc: [number, number];
	}
	interface TreStats {
		mean: number | null;
		median: number | null;
		max: number | null;
		p95: number | null;
		per_point: number[];
		error?: string;
	}
	interface TreResult {
		n: number;
		field_set_id: string | null;
		none: TreStats;
		regwsi: TreStats;
		ours: TreStats;
	}

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
	let downX = 0;
	let downY = 0;

	let emphasis = $state<'he' | 'ihc' | null>('he');
	let showGrid = $state(true);
	let phase = $state<'he' | 'ihc'>('he');
	let pendingHe = $state<[number, number] | null>(null);
	let landmarks = $state<Landmark[]>([]);
	let saveBusy = $state(false);
	let tre = $state<TreResult | null>(null);
	let treBusy = $state(false);
	let treErr = $state<string | null>(null);
	let loaded = $state(0);

	const heOpacity = $derived(emphasis === 'ihc' ? 0 : 1);
	const ihcOpacity = $derived(emphasis === 'he' ? 0 : emphasis === 'ihc' ? 1 : 0.5);
	const quads = $derived(
		Array.from({ length: nq * nq }, (_, i) => ({ qy: Math.floor(i / nq), qx: i % nq }))
	);
	const totalImgs = $derived(nq * nq * 2);
	const allLoaded = $derived(loaded >= totalImgs);
	const pairNum = $derived(landmarks.length + 1);

	const cellW = $derived(imgW > 0 ? imgW / GRID : CNN_WIDTH);
	const cellH = $derived(imgH > 0 ? imgH / GRID : CNN_HEIGHT);
	const vLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellW));
	const hLines = $derived(Array.from({ length: GRID + 1 }, (_, i) => i * cellH));

	function quadSrc(layer: 'he' | 'ihc', qy: number, qx: number) {
		return `/api/regwsi/full?pair=${pairId}&layer=${layer}&qy=${qy}&qx=${qx}`;
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

	function setPhase(next: 'he' | 'ihc') {
		phase = next;
		emphasis = next;
	}

	async function loadLandmarks() {
		const r = await fetch(`/api/regwsi/landmarks?pair=${pairId}`);
		if (!r.ok) return;
		const d = await r.json();
		landmarks = d.points ?? [];
		pendingHe = null;
		setPhase('he');
		if (landmarks.length > 0) await refreshTre();
		else {
			tre = null;
			treErr = null;
		}
	}

	async function persistLandmarks(next: Landmark[]) {
		saveBusy = true;
		try {
			const r = await fetch(`/api/regwsi/landmarks?pair=${pairId}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ points: next })
			});
			if (r.ok) {
				const d = await r.json();
				landmarks = d.points ?? next;
				await refreshTre();
			}
		} finally {
			saveBusy = false;
		}
	}

	async function clearLandmarks() {
		pendingHe = null;
		const r = await fetch(`/api/regwsi/landmarks?pair=${pairId}`, { method: 'DELETE' });
		if (r.ok) {
			landmarks = [];
			tre = null;
			treErr = null;
			setPhase('he');
		}
	}

	async function undoLast() {
		if (pendingHe) {
			pendingHe = null;
			setPhase('he');
			return;
		}
		if (!landmarks.length) return;
		const next = landmarks.slice(0, -1);
		await persistLandmarks(next);
		setPhase('he');
	}

	async function refreshTre() {
		if (!landmarks.length) {
			tre = null;
			treErr = null;
			return;
		}
		treBusy = true;
		treErr = null;
		try {
			const r = await fetch(`/api/regwsi/tre?pair=${pairId}`);
			if (!r.ok) throw new Error(await r.text());
			tre = await r.json();
		} catch (e) {
			treErr = e instanceof Error ? e.message : 'tre failed';
			tre = null;
		} finally {
			treBusy = false;
		}
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

	async function onFrameClick(e: PointerEvent) {
		if (dragging || saveBusy) return;
		const pt = clientToNorm(e);
		if (!pt) return;
		if (phase === 'he') {
			pendingHe = pt;
			setPhase('ihc');
			return;
		}
		if (!pendingHe) {
			setPhase('he');
			return;
		}
		const next = [...landmarks, { he: pendingHe, ihc: pt }];
		pendingHe = null;
		await persistLandmarks(next);
		setPhase('he');
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
		void loadLandmarks();
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
		if (moved < 5) void onFrameClick(e);
	}

	function fmt(v: number | null | undefined) {
		return v == null ? '—' : v.toFixed(2);
	}
</script>

<div class="page">
	<header class="head">
		<div>
			<h1>Annotate · pair {pairId}</h1>
			<p class="sub">
				HE → raw IHC correspondences for TRE (L5 px) vs {data.mainSetName ?? 'main field set'}.
				Shift+Q/W solo HE / IHC. No warped IHC on this page.
			</p>
		</div>
		<nav class="nav">
			<a href={`/regwsi/${pairId}`}>Overlay</a>
			<a href="/regwsi">All pairs</a>
		</nav>
	</header>

	{#if !data.fullReady || !data.fullMeta}
		<div class="empty">
			No HE + raw IHC mosaic for this pair yet. Run:
			<code>python regWSI/make_full.py {pairId}</code>
			{#if data.ready}
				<span class="hint">(registration exists; mosaic only)</span>
			{/if}
		</div>
	{:else}
		<div class="controls">
			<span class="phase" class:ihc={phase === 'ihc'}>
				{#if phase === 'he'}
					Pair {pairNum} · click HE
				{:else}
					Pair {pairNum} · click raw IHC match
				{/if}
				· {landmarks.length} saved
			</span>
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
					title="Show only raw IHC (Shift+W); press again for overlay"
				>IHC</button>
			</span>
			<label class="chk">
				<input type="checkbox" bind:checked={showGrid} />
				L5 grid
			</label>
			<button class="btn" onclick={undoLast} disabled={saveBusy}>Undo</button>
			<button class="btn" onclick={clearLandmarks} disabled={saveBusy}>Clear</button>
			<button class="reset" onclick={fit} disabled={!allLoaded}>Reset view</button>
			<span class="zoom">{allLoaded ? `${(scale / baseScale).toFixed(1)}×` : `${loaded}/${totalImgs}`}</span>
		</div>

		<div class="body">
			<div
				class="frame annotating"
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
					{#key `${pairId}-${nq}`}
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
					<svg class="marks" width={imgW} height={imgH} viewBox={`0 0 ${imgW} ${imgH}`} aria-hidden="true">
						{#each landmarks as lm, i}
							{@const hx = lm.he[0] * imgW}
							{@const hy = lm.he[1] * imgH}
							{@const ix = lm.ihc[0] * imgW}
							{@const iy = lm.ihc[1] * imgH}
							<circle class="mk he" cx={hx} cy={hy} r={6 / scale} />
							<text class="mk-label" x={hx + 8 / scale} y={hy - 8 / scale} style:font-size={`${12 / scale}px`}>{i + 1}</text>
							<circle class="mk ihc" cx={ix} cy={iy} r={6 / scale} />
							<line class="mk-link" x1={hx} y1={hy} x2={ix} y2={iy} />
						{/each}
						{#if pendingHe}
							<circle
								class="mk he pending"
								cx={pendingHe[0] * imgW}
								cy={pendingHe[1] * imgH}
								r={7 / scale}
							/>
						{/if}
					</svg>
				</div>
				{#if !allLoaded}
					<span class="loading">loading {loaded}/{totalImgs}…</span>
				{/if}
			</div>

			<aside class="tre-panel">
				<h2>TRE</h2>
				<p class="tre-sub">
					L5 px · vs {data.mainSetName ?? '—'}
					{#if data.mainSetId}<span class="muted">({data.mainSetId})</span>{/if}
				</p>
				{#if treBusy}
					<p class="muted">Computing…</p>
				{:else if landmarks.length === 0}
					<p class="muted">Add correspondences to compute TRE</p>
				{:else if treErr}
					<p class="err">{treErr}</p>
				{:else if tre}
					<table>
						<thead>
							<tr>
								<th></th>
								<th>none</th>
								<th>regWSI</th>
								<th>ours</th>
							</tr>
						</thead>
						<tbody>
							<tr>
								<td>mean L5 px</td>
								<td>{fmt(tre.none.mean)}</td>
								<td>{fmt(tre.regwsi.mean)}</td>
								<td>{fmt(tre.ours.mean)}</td>
							</tr>
							<tr>
								<td>median</td>
								<td>{fmt(tre.none.median)}</td>
								<td>{fmt(tre.regwsi.median)}</td>
								<td>{fmt(tre.ours.median)}</td>
							</tr>
							<tr>
								<td>max</td>
								<td>{fmt(tre.none.max)}</td>
								<td>{fmt(tre.regwsi.max)}</td>
								<td>{fmt(tre.ours.max)}</td>
							</tr>
							<tr>
								<td>p95</td>
								<td>{fmt(tre.none.p95)}</td>
								<td>{fmt(tre.regwsi.p95)}</td>
								<td>{fmt(tre.ours.p95)}</td>
							</tr>
						</tbody>
					</table>
					{#if tre.ours.error}
						<p class="err">{tre.ours.error}</p>
					{/if}
					{#if tre.none.per_point?.length}
						<ul class="per">
							{#each tre.none.per_point as e, i}
								<li>
									#{i + 1}
									<span>n {e.toFixed(1)}</span>
									<span>r {(tre.regwsi.per_point[i] ?? NaN).toFixed(1)}</span>
									<span>o {(tre.ours.per_point[i] ?? NaN).toFixed(1)}</span>
								</li>
							{/each}
						</ul>
					{/if}
				{/if}
			</aside>
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
	.phase {
		padding: 0.25rem 0.6rem;
		border-radius: 3px;
		background: #dbeafe;
		color: #1e3a8a;
		font-weight: 600;
		font-size: 0.8rem;
	}
	.phase.ihc {
		background: #fef3c7;
		color: #92400e;
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
	.btn {
		padding: 0.3rem 0.65rem;
		border: 1px solid #ccc;
		border-radius: 3px;
		background: #fff;
		cursor: pointer;
		font-size: 0.8rem;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: default;
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
	.body {
		flex: 1;
		min-height: 0;
		display: grid;
		grid-template-columns: 1fr 260px;
		gap: 0.75rem;
	}
	.frame {
		position: relative;
		min-height: 320px;
		border: 1px solid #ddd;
		background: #1a1a1a;
		overflow: hidden;
		cursor: crosshair;
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
	.grid,
	.marks {
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
	.mk {
		fill: none;
		stroke-width: 2;
		vector-effect: non-scaling-stroke;
	}
	.mk.he {
		stroke: #3b82f6;
	}
	.mk.ihc {
		stroke: #f59e0b;
	}
	.mk.pending {
		stroke: #3b82f6;
		stroke-dasharray: 4 3;
	}
	.mk-link {
		stroke: rgba(255, 255, 255, 0.35);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.mk-label {
		fill: #3b82f6;
		font-weight: 700;
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
	.tre-panel {
		border: 1px solid #ddd;
		border-radius: 4px;
		padding: 0.75rem;
		overflow: auto;
		font-size: 0.8rem;
	}
	.tre-panel h2 {
		margin: 0 0 0.25rem;
		font-size: 0.95rem;
	}
	.tre-sub {
		margin: 0 0 0.75rem;
		color: #666;
		font-size: 0.75rem;
		line-height: 1.3;
	}
	.muted {
		color: #999;
	}
	.tre-panel table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 0.5rem;
	}
	.tre-panel th,
	.tre-panel td {
		text-align: right;
		padding: 0.2rem 0.25rem;
		border-bottom: 1px solid #eee;
	}
	.tre-panel th:first-child,
	.tre-panel td:first-child {
		text-align: left;
		color: #666;
	}
	.err {
		color: #b91c1c;
		font-size: 0.75rem;
		margin: 0.5rem 0 0;
	}
	.per {
		list-style: none;
		margin: 0.75rem 0 0;
		padding: 0;
		max-height: 12rem;
		overflow: auto;
	}
	.per li {
		display: flex;
		gap: 0.5rem;
		justify-content: space-between;
		padding: 0.15rem 0;
		border-bottom: 1px solid #f0f0f0;
		font-variant-numeric: tabular-nums;
		color: #555;
	}
</style>
