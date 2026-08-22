<script lang="ts">
	export interface TreStats {
		mean: number | null;
		median: number | null;
		max: number | null;
		p95: number | null;
		per_point?: number[];
		error?: string;
	}

	export interface MethodCell {
		key: string;
		lam: string;
		field_estimator: string;
		complete: boolean;
		tre: TreStats;
		runtime_s?: number | null;
		runtime_avg_s?: number | null;
	}

	export interface BatchConfigSummary {
		wendland_eps?: number;
		wendland_eps_by_lam?: Record<string, number>;
		eps_label?: string;
		bspline_grid?: number;
		bspline_reg?: number;
		gate?: string;
		fingerprint?: string;
	}

	export interface BatchTreResult {
		pair_id: number;
		batch_id: string;
		n: number;
		none: TreStats;
		regwsi: TreStats;
		regwsi_runtime_s?: number | null;
		regwsi_runtime_avg_s?: number | null;
		methods: MethodCell[];
		config?: BatchConfigSummary;
	}

	let {
		pairId,
		tre,
		treBusy,
		treErr,
		landmarkCount,
		rigidInliers = null,
		batchId = null,
		dataset = 'muromi',
		onClose = undefined
	}: {
		pairId: number | null;
		tre: BatchTreResult | null;
		treBusy: boolean;
		treErr: string | null;
		landmarkCount: number;
		rigidInliers?: { nInliers: number; nTotal: number; inlierPx: number | null } | null;
		batchId?: string | null;
		dataset?: 'muromi' | 'acrobat' | 'anhir';
		onClose?: (() => void) | undefined;
	} = $props();

	function fmt(v: number | null | undefined) {
		return v == null ? '—' : v.toFixed(2);
	}

	function fmtSec(v: number | null | undefined) {
		if (v == null || Number.isNaN(v)) return '—';
		if (v < 10) return `${v.toFixed(2)}s`;
		if (v < 60) return `${v.toFixed(1)}s`;
		const m = Math.floor(v / 60);
		const s = v - m * 60;
		return `${m}m${s.toFixed(0)}s`;
	}

	function overlayQuery() {
		const q = new URLSearchParams({ dataset });
		if (batchId) q.set('batch', batchId);
		return q;
	}

	function openRegwsi() {
		if (pairId == null) return;
		window.open(`/eval/${pairId}/overlay/regwsi?${overlayQuery()}`, `eval-overlay-${pairId}`);
	}

	function openNative() {
		if (pairId == null) return;
		const q = overlayQuery();
		window.open(`/eval/${pairId}/native/he?${q}`, `eval-native-${pairId}-he`);
		window.open(`/eval/${pairId}/native/ihc?${q}`, `eval-native-${pairId}-ihc`);
	}

	function openMethod(m: MethodCell) {
		if (pairId == null || !batchId) return;
		const q = new URLSearchParams({
			estimator: m.field_estimator,
			lam: m.lam,
			batch: batchId,
			dataset
		});
		window.open(`/eval/${pairId}/overlay/fieldset?${q}`, `eval-overlay-${pairId}`);
	}

	const methods = $derived(tre?.methods ?? []);
	const cfg = $derived(tre?.config ?? null);

	let cpuRt = $state<{
		hosts: Record<
			string,
			{
				threads?: number;
				methods?: Record<string, { runtime_s?: number }>;
			}
		>;
	} | null>(null);

	$effect(() => {
		if (pairId == null) {
			cpuRt = null;
			return;
		}
		const p = pairId;
		const ds = dataset;
		void fetch(`/api/eval/cpu-runtime?pair=${p}&dataset=${ds}`)
			.then((r) => (r.ok ? r.json() : null))
			.then((j) => {
				if (p !== pairId) return;
				cpuRt = j;
			})
			.catch(() => {
				if (p === pairId) cpuRt = null;
			});
	});

	const cpuHosts = $derived.by(() => {
		const h = cpuRt?.hosts ?? {};
		return ['vps', 'm4'].filter((k) => h[k]);
	});
	const cpuMethods = ['regwsi', 'fft', 'superpoint_glue'] as const;
	const cpuLabels: Record<(typeof cpuMethods)[number], string> = {
		regwsi: 'regWSI',
		fft: 'FFT',
		superpoint_glue: 'SP+LG'
	};

	function cpuSec(host: string, method: string): number | null {
		const v = cpuRt?.hosts?.[host]?.methods?.[method]?.runtime_s;
		return typeof v === 'number' ? v : null;
	}
</script>

<aside class="tre-panel">
	<div class="head">
		<h2>
			TRE{#if pairId != null}
				<span class="pair"> · pair {pairId}</span>
			{/if}
		</h2>
		{#if onClose}
			<button type="button" class="close" onclick={onClose} title="Close">×</button>
		{/if}
	</div>
	<p class="tre-sub">
		L5 px{#if batchId}
			· batch {batchId}{/if}
	</p>
	{#if rigidInliers}
		<p class="inliers" class:bad={rigidInliers.nInliers === 0}>
			Rigid inliers {rigidInliers.nInliers}/{rigidInliers.nTotal}
			({((100 * rigidInliers.nInliers) / rigidInliers.nTotal).toFixed(rigidInliers.nInliers / rigidInliers.nTotal < 0.01 ? 1 : 0)}%)
			{#if rigidInliers.inlierPx != null}
				<span class="hint"> · residual ≤ {Math.round(rigidInliers.inlierPx)} canvas px</span>
			{/if}
		</p>
	{/if}

	{#if treBusy}
		<p class="muted">Computing…</p>
	{:else if landmarkCount === 0}
		<p class="muted">Add landmarks to compute TRE</p>
	{:else if treErr}
		<p class="err">{treErr}</p>
	{:else if tre}
		{#if cfg}
			<div class="cfg">
				<span>{cfg.eps_label ?? `Wendland ε=${cfg.wendland_eps ?? '—'}`}</span>
				<span>B-spline grid={cfg.bspline_grid ?? '—'} reg={cfg.bspline_reg ?? '—'}</span>
				<span>{cfg.gate ?? ''}</span>
				{#if cfg.fingerprint}
					<span class="fp">cfg {cfg.fingerprint}</span>
				{/if}
			</div>
		{/if}
		<div class="scroll">
			<table>
				<thead>
					<tr>
						<th></th>
						<th>none</th>
						<th>regWSI</th>
						{#each methods as m}
							<th title={m.key}>{m.lam.slice(0, 3)}/{m.field_estimator.slice(0, 3)}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					<tr>
						<td>mean</td>
						<td>{fmt(tre.none.mean)}</td>
						<td>{fmt(tre.regwsi.mean)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.mean)}</td>
						{/each}
					</tr>
					<tr>
						<td>median</td>
						<td>{fmt(tre.none.median)}</td>
						<td>{fmt(tre.regwsi.median)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.median)}</td>
						{/each}
					</tr>
					<tr>
						<td>max</td>
						<td>{fmt(tre.none.max)}</td>
						<td>{fmt(tre.regwsi.max)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.max)}</td>
						{/each}
					</tr>
					<tr>
						<td>p95</td>
						<td>{fmt(tre.none.p95)}</td>
						<td>{fmt(tre.regwsi.p95)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.p95)}</td>
						{/each}
					</tr>
					<tr class="runtime">
						<td>runtime</td>
						<td>—</td>
						<td>{fmtSec(tre.regwsi_runtime_s)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmtSec(m.runtime_s)}</td>
						{/each}
					</tr>
					<tr class="runtime">
						<td>runtime avg</td>
						<td>—</td>
						<td>{fmtSec(tre.regwsi_runtime_avg_s)}</td>
						{#each methods as m}
							<td>{fmtSec(m.runtime_avg_s)}</td>
						{/each}
					</tr>
				</tbody>
			</table>
		</div>
		<p class="n">n = {tre.n} landmarks · runtime = this pair · avg = mean over batch pairs with a stored time</p>
		{#if cpuHosts.length}
			<div class="cpu-rt">
				<p class="cpu-h">CPU wall clock</p>
				<table>
					<thead>
						<tr>
							<th></th>
							{#each cpuMethods as m}
								<th>{cpuLabels[m]}</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						{#each cpuHosts as host}
							<tr class="runtime">
								<td>{host}</td>
								{#each cpuMethods as m}
									<td>{fmtSec(cpuSec(host, m))}</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
		<div class="actions">
			<button type="button" class="btn" disabled={tre.regwsi.mean == null} onclick={openRegwsi}>
				Open regWSI overlay
			</button>
			{#each methods as m}
				<button
					type="button"
					class="btn"
					disabled={!m.complete || m.tre.mean == null || !batchId}
					onclick={() => openMethod(m)}
					title={m.key}
				>
					Open {m.lam}/{m.field_estimator}
				</button>
			{/each}
		</div>
	{/if}
	{#if pairId != null}
		<div class="actions">
			<button type="button" class="btn" onclick={openNative}>Open native HE + IHC</button>
		</div>
	{/if}
</aside>

<style>
	.tre-panel {
		display: flex;
		flex-direction: column;
		gap: 0.65rem;
		min-height: 0;
		height: 100%;
		padding: 0.85rem 1rem;
		background: #12141a;
		border-left: 1px solid #2a2d3a;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	h2 {
		margin: 0;
		font-size: 0.95rem;
		font-weight: 600;
		color: #e8eaf0;
	}
	.pair {
		color: #9ca3af;
		font-weight: 400;
	}
	.close {
		border: none;
		background: transparent;
		color: #9ca3af;
		font-size: 1.2rem;
		cursor: pointer;
		line-height: 1;
	}
	.tre-sub {
		margin: 0;
		font-size: 0.72rem;
		color: #6b7280;
	}
	.inliers {
		margin: 0;
		font-size: 0.75rem;
		color: #86efac;
		font-variant-numeric: tabular-nums;
	}
	.inliers.bad {
		color: #f87171;
	}
	.inliers .hint {
		color: #6b7280;
		font-weight: 400;
	}
	.cfg {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem 0.75rem;
		font-size: 0.68rem;
		color: #9ca3af;
		line-height: 1.35;
	}
	.cfg .fp {
		color: #6b7280;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}
	.muted {
		margin: 0;
		color: #6b7280;
		font-size: 0.8rem;
	}
	.err {
		margin: 0;
		color: #f87171;
		font-size: 0.8rem;
	}
	.scroll {
		overflow: auto;
		flex: 1;
		min-height: 0;
	}
	table {
		border-collapse: collapse;
		font-size: 0.72rem;
		width: max-content;
		min-width: 100%;
	}
	th,
	td {
		padding: 0.28rem 0.45rem;
		border-bottom: 1px solid #1f2330;
		text-align: right;
		white-space: nowrap;
	}
	th:first-child,
	td:first-child {
		text-align: left;
		color: #9ca3af;
		position: sticky;
		left: 0;
		background: #12141a;
	}
	th {
		color: #9ca3af;
		font-weight: 500;
	}
	td.missing {
		color: #4b5563;
	}
	tr.runtime td {
		color: #a5b4fc;
	}
	.n {
		margin: 0;
		font-size: 0.72rem;
		color: #6b7280;
	}
	.cpu-h {
		margin: 0 0 0.35rem;
		font-size: 0.72rem;
		color: #9ca3af;
	}
	.cpu-rt {
		margin-top: 0.15rem;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
	}
	.btn {
		padding: 0.3rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		background: #181b23;
		color: #e8eaf0;
		font-size: 0.7rem;
		cursor: pointer;
	}
	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn:not(:disabled):hover {
		border-color: #3b82f6;
	}
</style>
