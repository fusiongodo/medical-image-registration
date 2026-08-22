<script lang="ts">
	type LamStats = {
		n: number;
		mean_of_means: number | null;
		mean_of_medians: number | null;
	};

	type Point = {
		eps: number;
		batch_id: string;
		n_pairs: number;
		status: { state?: string; done?: number; total?: number } | null;
		fft: LamStats;
		superpoint_glue: LamStats;
	};

	let { dataset }: { dataset: string } = $props();

	let points = $state<Point[]>([]);
	let poll: ReturnType<typeof setInterval> | null = null;

	async function load() {
		const r = await fetch(`/api/eval/eps-sweep?dataset=${dataset}`);
		if (!r.ok) return;
		const j = await r.json();
		points = Array.isArray(j.points) ? j.points : [];
	}

	$effect(() => {
		void dataset;
		void load();
		if (poll) clearInterval(poll);
		poll = setInterval(() => void load(), 4000);
		return () => {
			if (poll) clearInterval(poll);
			poll = null;
		};
	});

	function fmt(v: number | null | undefined) {
		return v == null ? '—' : v.toFixed(1);
	}

	function fmtEps(v: number) {
		return v.toFixed(v % 0.1 === 0 ? 1 : 2);
	}

	const series = $derived.by(() => {
		const xs = points.map((p) => p.eps);
		const fft = points.map((p) => p.fft.mean_of_means);
		const sp = points.map((p) => p.superpoint_glue.mean_of_means);
		const ys = [...fft, ...sp].filter((v): v is number => v != null);
		return { xs, fft, sp, ys };
	});

	const W = 420;
	const H = 140;
	const pad = { l: 36, r: 10, t: 10, b: 22 };

	function xOf(eps: number, xmin: number, xmax: number) {
		const span = Math.max(1e-6, xmax - xmin);
		return pad.l + ((eps - xmin) / span) * (W - pad.l - pad.r);
	}

	function yOf(v: number, ymin: number, ymax: number) {
		const span = Math.max(1e-6, ymax - ymin);
		return pad.t + (1 - (v - ymin) / span) * (H - pad.t - pad.b);
	}

	function linePath(
		vals: (number | null)[],
		xs: number[],
		xmin: number,
		xmax: number,
		y0: number,
		y1: number
	) {
		const parts: string[] = [];
		let started = false;
		for (let i = 0; i < vals.length; i++) {
			const v = vals[i];
			if (v == null) {
				started = false;
				continue;
			}
			parts.push(
				`${started ? 'L' : 'M'}${xOf(xs[i], xmin, xmax).toFixed(1)},${yOf(v, y0, y1).toFixed(1)}`
			);
			started = true;
		}
		return parts.join(' ');
	}

	const plot = $derived.by(() => {
		if (!series.ys.length) return null;
		const xmin = Math.min(...series.xs);
		const xmax = Math.max(...series.xs);
		const ymin = Math.min(...series.ys);
		const ymax = Math.max(...series.ys);
		const yPad = Math.max(2, (ymax - ymin) * 0.12);
		const y0 = ymin - yPad;
		const y1 = ymax + yPad;
		return {
			xmin,
			xmax,
			y0,
			y1,
			fft: linePath(series.fft, series.xs, xmin, xmax, y0, y1),
			sp: linePath(series.sp, series.xs, xmin, xmax, y0, y1),
			dots: points.flatMap((p) => {
				const x = xOf(p.eps, xmin, xmax);
				const out: { x: number; y: number; cls: string }[] = [];
				if (p.fft.mean_of_means != null) {
					out.push({ x, y: yOf(p.fft.mean_of_means, y0, y1), cls: 'fft' });
				}
				if (p.superpoint_glue.mean_of_means != null) {
					out.push({ x, y: yOf(p.superpoint_glue.mean_of_means, y0, y1), cls: 'sp' });
				}
				return out;
			})
		};
	});

	const running = $derived(
		points.some((p) => (p.status?.state || '').toLowerCase() === 'running')
	);
</script>

{#if points.length}
	<section class="eps">
		<div class="head">
			<h2>Wendland ε vs TRE</h2>
			<p class="sub">
				mean of per-pair mean L5 px · FFT × Wendland / SP+LG × Wendland
				{#if running}<span class="live">updating</span>{/if}
			</p>
		</div>
		<div class="body">
			{#if plot}
				<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="TRE versus Wendland epsilon">
					<text class="axis" x="4" y="12">{fmt(plot.y1)}</text>
					<text class="axis" x="4" y={H - 26}>{fmt(plot.y0)}</text>
					{#if plot.fft}
						<path class="line fft" d={plot.fft} />
					{/if}
					{#if plot.sp}
						<path class="line sp" d={plot.sp} />
					{/if}
					{#each plot.dots as d}
						<circle class={d.cls} cx={d.x} cy={d.y} r="3" />
					{/each}
					{#each points as p}
						<text class="axis" x={xOf(p.eps, plot.xmin, plot.xmax)} y={H - 6} text-anchor="middle"
							>{fmtEps(p.eps)}</text
						>
					{/each}
				</svg>
			{/if}
			<table>
				<thead>
					<tr>
						<th>ε</th>
						<th>batch</th>
						<th>FFT mean</th>
						<th>SP mean</th>
						<th>n</th>
					</tr>
				</thead>
				<tbody>
					{#each points as p}
						<tr>
							<td>{fmtEps(p.eps)}</td>
							<td class="id">{p.batch_id}</td>
							<td class:dim={p.fft.n === 0}>{fmt(p.fft.mean_of_means)}</td>
							<td class:dim={p.superpoint_glue.n === 0}>{fmt(p.superpoint_glue.mean_of_means)}</td>
							<td
								>{p.fft.n}/{p.n_pairs}{#if (p.status?.state || '') === 'running'}
									<span class="live"> · run</span>
								{/if}</td
							>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="legend">
			<span class="sw fft"></span> FFT
			<span class="sw sp"></span> SP+LG
		</p>
	</section>
{/if}

<style>
	.eps {
		margin: 0 0 1rem;
		padding: 0.7rem 0.85rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 6px;
		background: #12141a;
	}
	.head h2 {
		margin: 0;
		font-size: 0.82rem;
		font-weight: 600;
		color: #e8eaf0;
	}
	.sub {
		margin: 0.15rem 0 0;
		font-size: 0.7rem;
		color: #6b7280;
	}
	.body {
		display: grid;
		grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr);
		gap: 0.75rem;
		align-items: center;
		margin-top: 0.45rem;
	}
	.chart {
		width: 100%;
		height: auto;
		display: block;
	}
	.line {
		fill: none;
		stroke-width: 1.6;
	}
	.line.fft,
	circle.fft {
		stroke: #60a5fa;
		fill: #60a5fa;
	}
	.line.sp,
	circle.sp {
		stroke: #f59e0b;
		fill: #f59e0b;
	}
	.line.fft,
	.line.sp {
		fill: none;
	}
	.axis {
		fill: #6b7280;
		font-size: 9px;
	}
	table {
		border-collapse: collapse;
		font-size: 0.72rem;
		width: 100%;
	}
	th,
	td {
		padding: 0.2rem 0.35rem;
		text-align: right;
		border-bottom: 1px solid #1f2330;
		white-space: nowrap;
	}
	th:first-child,
	td:first-child,
	td.id {
		text-align: left;
	}
	th {
		color: #9ca3af;
		font-weight: 500;
	}
	td {
		color: #e8eaf0;
	}
	td.id {
		color: #9ca3af;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.68rem;
	}
	td.dim {
		color: #4b5563;
	}
	.legend {
		margin: 0.35rem 0 0;
		font-size: 0.68rem;
		color: #9ca3af;
		display: flex;
		gap: 0.7rem;
		align-items: center;
	}
	.sw {
		width: 0.65rem;
		height: 0.2rem;
		display: inline-block;
		margin-right: 0.25rem;
		border-radius: 1px;
	}
	.sw.fft {
		background: #60a5fa;
	}
	.sw.sp {
		background: #f59e0b;
	}
	.live {
		color: #f59e0b;
		font-size: 0.65rem;
	}
	@media (max-width: 720px) {
		.body {
			grid-template-columns: 1fr;
		}
	}
</style>
