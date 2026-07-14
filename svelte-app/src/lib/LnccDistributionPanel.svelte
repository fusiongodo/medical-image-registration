<script lang="ts">
	let {
		pairId,
		depth,
		patchSize = 50
	}: {
		pairId: number;
		depth: number;
		patchSize?: number;
	} = $props();

	interface DistData {
		bins: number[];
		counts: number[];
		maxVal: number;
		totalTiles: number;
		withDisplacement: number;
		withMetrics: number;
		missing: number;
	}

	interface JobState {
		running: boolean;
		done: number;
		total: number;
		error: string | null;
		finishedAt: number | null;
	}

	let open      = $state(true);
	let dist      = $state<DistData | null>(null);
	let job       = $state<JobState | null>(null);
	let pollTimer = $state<ReturnType<typeof setInterval> | null>(null);

	$effect(() => {
		try {
			const stored = localStorage.getItem('mvrLnccOpen');
			if (stored !== null) open = stored === '1';
		} catch {
			/* ignore storage errors */ }
	});

	$effect(() => {
		try {
			localStorage.setItem('mvrLnccOpen', open ? '1' : '0');
		} catch {
			/* ignore storage errors */ }
	});

	async function fetchDist() {
		const r = await fetch(`/api/lncc-distribution?pair=${pairId}&depth=${depth}&patchSize=${patchSize}`);
		dist = await r.json();
	}

	async function startCompute() {
		const r = await fetch('/api/lncc-distribution/compute', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: pairId, depth }),
		});
		const data = await r.json();
		job = data.state;
		if (job?.running) startPolling();
	}

	function startPolling() {
		if (pollTimer) return;
		pollTimer = setInterval(async () => {
			const r = await fetch(`/api/lncc-distribution/progress?pair=${pairId}&depth=${depth}`);
			job = await r.json();
			if (!job?.running) {
				clearInterval(pollTimer!);
				pollTimer = null;
				await fetchDist();
			}
		}, 1000);
	}

	$effect(() => {
		if (open && dist === null) fetchDist();
		return () => {
			if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
		};
	});

	// Recompute on pair/depth/patchSize change
	$effect(() => {
		void pairId; void depth; void patchSize;
		dist = null;
		job = null;
		if (open) fetchDist();
	});

	const SVG_W = 860;
	const SVG_H = 240;
	const PAD_L = 44;
	const PAD_B = 24;
	const PAD_T = 10;
	const PAD_R = 12;

	const chartW = SVG_W - PAD_L - PAD_R;
	const chartH = SVG_H - PAD_B - PAD_T;

	const bars = $derived.by(() => {
		if (!dist || dist.bins.length === 0) return [];
		const maxCount = Math.max(1, ...dist.counts);
		const bw = chartW / dist.bins.length;
		return dist.bins.map((bin, i) => ({
			x: PAD_L + i * bw,
			y: PAD_T + chartH - (dist!.counts[i] / maxCount) * chartH,
			w: Math.max(1, bw - 1),
			h: (dist!.counts[i] / maxCount) * chartH,
			label: bin.toFixed(2),
			count: dist!.counts[i],
		}));
	});

	const xTicks = $derived.by(() => {
		if (!dist || dist.bins.length === 0) return [];
		const maxVal = dist.maxVal;
		const N_TICKS = 11;
		return Array.from({ length: N_TICKS }, (_, k) => {
			const frac = k / (N_TICKS - 1);
			const v    = maxVal * frac;
			const decimals = v < 0.01 ? 4 : v < 0.1 ? 3 : v < 10 ? 2 : 0;
			return {
				x: PAD_L + frac * chartW,
				label: v.toFixed(decimals),
			};
		});
	});

	const yTicks = $derived.by(() => {
		if (!dist || dist.counts.length === 0) return [];
		const maxCount = Math.max(1, ...dist.counts);
		const step = maxCount <= 5 ? 1
			: maxCount <= 20 ? 5
			: maxCount <= 50 ? 10
			: maxCount <= 200 ? 50
			: maxCount <= 500 ? 100
			: Math.pow(10, Math.floor(Math.log10(maxCount)));
		const ticks: { y: number; label: string }[] = [];
		for (let v = 0; v <= maxCount; v += step) {
			const y = PAD_T + chartH - (v / maxCount) * chartH;
			ticks.push({ y, label: String(v) });
		}
		return ticks;
	});
</script>

<div class="panel">
	<button class="toggle" onclick={() => open = !open}>
		<span class="arrow">{open ? '▾' : '▸'}</span>
		LNCC² distribution
		{#if dist}
			<span class="summary-inline">
				· {dist.withMetrics} scored / {dist.withDisplacement} aligned
				{#if dist.missing > 0}<span class="missing">· {dist.missing} missing</span>{/if}
			</span>
		{/if}
	</button>

	{#if open}
		<div class="body">
			{#if dist === null}
				<span class="loading">loading…</span>
			{:else}
				{#if dist.withMetrics === 0}
					<div class="no-data">No scored tiles yet — click "Compute missing" to start.</div>
				{:else}
					<svg width={SVG_W} height={SVG_H} class="chart">
						{#each bars as b}
							<rect x={b.x} y={b.y} width={b.w} height={b.h} class="bar" />
						{/each}

						<line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + chartH} class="axis" />
						<line x1={PAD_L} y1={PAD_T + chartH} x2={PAD_L + chartW} y2={PAD_T + chartH} class="axis" />

						{#each xTicks as t}
							<line x1={t.x} y1={PAD_T + chartH} x2={t.x} y2={PAD_T + chartH + 4} class="tick" />
							<text x={t.x} y={SVG_H - 2} class="tick-label" text-anchor="middle">{t.label}</text>
						{/each}

						{#each yTicks as t}
							<line x1={PAD_L - 4} y1={t.y} x2={PAD_L} y2={t.y} class="tick" />
							<text x={PAD_L - 7} y={t.y + 4} class="tick-label" text-anchor="end">{t.label}</text>
						{/each}
					</svg>
				{/if}

				<div class="controls">
					{#if dist.missing > 0}
						{#if job?.running}
							<span class="progress">
								Computing… {job.done} / {job.total || '?'} tiles
							</span>
						{:else}
							<button class="compute-btn" onclick={startCompute}>
								Compute missing ({dist.missing})
							</button>
						{/if}
						{#if job?.error}
							<span class="err">{job.error}</span>
						{/if}
					{:else}
						<span class="all-done">All {dist.withMetrics} tiles scored</span>
					{/if}
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.panel {
		border-bottom: 1px solid #2a2d3a;
		background: #131520;
		flex-shrink: 0;
	}

	.toggle {
		all: unset;
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 14px;
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: #6b7280;
		width: 100%;
	}

	.toggle:hover { color: #e8eaf0; }

	.arrow { font-size: 0.65rem; }

	.summary-inline {
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
		color: #9ca3af;
	}

	.missing { color: #f59e0b; }

	.body {
		padding: 8px 14px 12px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.loading { font-size: 0.8rem; color: #6b7280; padding: 12px 0; }

	.no-data { font-size: 0.8rem; color: #6b7280; padding: 12px 0; }

	.chart { display: block; max-width: 100%; }

	.bar { fill: #6366f1; opacity: 0.85; }

	.axis { stroke: #4b5563; stroke-width: 1; }

	.tick { stroke: #4b5563; stroke-width: 1; }

	.tick-label {
		fill: #9ca3af;
		font-size: 11px;
		font-family: system-ui, sans-serif;
	}

	.controls {
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 12px;
	}

	.compute-btn {
		all: unset;
		cursor: pointer;
		background: #6366f1;
		color: #fff;
		font-size: 0.72rem;
		font-weight: 600;
		padding: 5px 12px;
		border-radius: 4px;
		white-space: nowrap;
	}

	.compute-btn:hover { background: #4f51c8; }

	.progress {
		font-size: 0.72rem;
		color: #f59e0b;
		white-space: nowrap;
	}

	.all-done {
		font-size: 0.72rem;
		color: #22c55e;
		white-space: nowrap;
	}

	.err {
		font-size: 0.7rem;
		color: #ef4444;
		white-space: nowrap;
	}
</style>
