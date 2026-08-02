<script lang="ts">
	import {
		connectSession,
		emptySession,
		readStoredSession,
		type AnnotateSession,
		type Landmark
	} from '$lib/regwsi/annotateSession';

	let { data } = $props();
	const pairId = $derived(data.pairId);

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

	let session = $state<AnnotateSession>(emptySession());
	let saveBusy = $state(false);
	let tre = $state<TreResult | null>(null);
	let treBusy = $state(false);
	let treErr = $state<string | null>(null);
	let handle: ReturnType<typeof connectSession> | null = null;

	function applyRemote(remote: AnnotateSession) {
		if (remote.rev > session.rev) {
			session = remote;
			void refreshTre();
		}
	}

	async function loadFromServer() {
		const r = await fetch(`/api/regwsi/landmarks?pair=${pairId}`);
		if (!r.ok) return;
		const d = await r.json();
		const landmarks = (d.points ?? []) as Landmark[];
		const stored = readStoredSession(pairId);
		if (stored && stored.rev > 0) {
			session = {
				phase: stored.phase,
				pendingHe: stored.pendingHe,
				landmarks,
				rev: stored.rev
			};
		} else {
			session = emptySession(landmarks);
			if (handle) {
				session = handle.publish(emptySession(landmarks), {
					landmarks,
					pendingHe: null,
					phase: 'he'
				});
			}
		}
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
				return (d.points ?? next) as Landmark[];
			}
		} finally {
			saveBusy = false;
		}
		return next;
	}

	async function undoLast() {
		if (!handle || saveBusy) return;
		if (session.pendingHe) {
			session = handle.publish(session, { pendingHe: null, phase: 'he' });
			return;
		}
		if (!session.landmarks.length) return;
		const next = session.landmarks.slice(0, -1);
		const saved = await persistLandmarks(next);
		session = handle.publish(session, { landmarks: saved, pendingHe: null, phase: 'he' });
		await refreshTre();
	}

	async function clearLandmarks() {
		if (!handle || saveBusy) return;
		const r = await fetch(`/api/regwsi/landmarks?pair=${pairId}`, { method: 'DELETE' });
		if (!r.ok) return;
		session = handle.publish(session, { landmarks: [], pendingHe: null, phase: 'he' });
		tre = null;
		treErr = null;
	}

	async function refreshTre() {
		if (!session.landmarks.length) {
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

	function openSide(side: 'he' | 'ihc') {
		window.open(`/regwsi/${pairId}/annotate/${side}`, `regwsi-annotate-${pairId}-${side}`);
	}

	function fmt(v: number | null | undefined) {
		return v == null ? '—' : v.toFixed(2);
	}

	$effect(() => {
		void pairId;
		handle?.close();
		handle = connectSession(pairId, applyRemote);
		void loadFromServer();
		return () => {
			handle?.close();
			handle = null;
		};
	});
</script>

<div class="page">
	<header class="head">
		<div>
			<h1>Annotate · pair {pairId}</h1>
			<p class="sub">
				Open HE and IHC in separate windows. Alternating clicks are enforced across windows —
				you cannot place two points in a row on the same side.
			</p>
		</div>
		<nav class="nav">
			<a href={`/regwsi/${pairId}`}>Overlay</a>
			<a href="/regwsi">All pairs</a>
		</nav>
	</header>

	{#if !data.fullReady || !data.fullMeta}
		<div class="empty">
			No HE + raw IHC mosaic for this pair yet.
			<code>python regWSI/make_full.py {pairId} --layers he ihc</code>
			{#if data.ready}
				<span class="hint">(registration exists; mosaic only)</span>
			{/if}
		</div>
	{:else}
		<section class="launch">
			<button type="button" class="open he" onclick={() => openSide('he')}>Open HE</button>
			<button type="button" class="open ihc" onclick={() => openSide('ihc')}>Open IHC</button>
		</section>

		<div class="status" class:ihc={session.phase === 'ihc'}>
			{#if session.phase === 'he'}
				Next click · HE window
			{:else}
				Next click · IHC window
			{/if}
			<span class="meta"
				>· {session.landmarks.length} saved{#if session.pendingHe} · HE pending{/if}</span
			>
		</div>

		<div class="toolbar">
			<button class="btn" onclick={undoLast} disabled={saveBusy}>Undo</button>
			<button class="btn" onclick={clearLandmarks} disabled={saveBusy}>Clear</button>
			<button class="btn" onclick={() => refreshTre()} disabled={treBusy || !session.landmarks.length}
				>Refresh TRE</button
			>
		</div>

		<aside class="tre-panel">
			<h2>TRE</h2>
			<p class="tre-sub">
				L5 px · vs {data.mainSetName ?? '—'}
				{#if data.mainSetId}<span class="muted">({data.mainSetId})</span>{/if}
			</p>
			{#if treBusy}
				<p class="muted">Computing…</p>
			{:else if session.landmarks.length === 0}
				<p class="muted">Add correspondences in the HE / IHC windows</p>
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
	{/if}
</div>

<style>
	.page {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
		overflow-y: auto;
		padding: 1.25rem 1.5rem 2rem;
		box-sizing: border-box;
		gap: 0.85rem;
		max-width: 40rem;
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
		color: #9ca3af;
		font-size: 0.85rem;
		line-height: 1.4;
	}
	.nav {
		display: flex;
		gap: 0.75rem;
		font-size: 0.85rem;
		white-space: nowrap;
	}
	.nav a {
		color: #c4c9d4;
	}
	.empty {
		padding: 1.5rem;
		border: 1px dashed #2a2d3a;
		border-radius: 4px;
		color: #9ca3af;
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.empty code {
		display: block;
		margin-top: 0.5rem;
		padding: 0.5rem 0.65rem;
		background: #181b23;
		border-radius: 3px;
		font-size: 0.8rem;
		color: #e8eaf0;
	}
	.empty .hint {
		display: block;
		margin-top: 0.5rem;
		color: #6b7280;
		font-size: 0.8rem;
	}
	.launch {
		display: flex;
		gap: 0.75rem;
	}
	.open {
		flex: 1;
		padding: 1rem 1.1rem;
		border-radius: 6px;
		border: 1px solid #2a2d3a;
		font-size: 1rem;
		font-weight: 700;
		letter-spacing: 0.04em;
		cursor: pointer;
	}
	.open.he {
		background: #1e3a5f;
		color: #bfdbfe;
		border-color: #3b82f6;
	}
	.open.ihc {
		background: #5c3d0e;
		color: #fde68a;
		border-color: #f59e0b;
	}
	.open:hover {
		filter: brightness(1.08);
	}
	.status {
		padding: 0.45rem 0.75rem;
		border-radius: 4px;
		background: #1e3a5f;
		color: #bfdbfe;
		font-weight: 600;
		font-size: 0.85rem;
	}
	.status.ihc {
		background: #5c3d0e;
		color: #fde68a;
	}
	.status .meta {
		font-weight: 500;
		opacity: 0.85;
	}
	.toolbar {
		display: flex;
		gap: 0.5rem;
	}
	.btn {
		padding: 0.35rem 0.7rem;
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
	.tre-panel {
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		padding: 0.85rem;
		background: #181b23;
		font-size: 0.8rem;
	}
	.tre-panel h2 {
		margin: 0 0 0.25rem;
		font-size: 0.95rem;
	}
	.tre-sub {
		margin: 0 0 0.75rem;
		color: #9ca3af;
		font-size: 0.75rem;
		line-height: 1.3;
	}
	.muted {
		color: #6b7280;
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
		border-bottom: 1px solid #2a2d3a;
	}
	.tre-panel th:first-child,
	.tre-panel td:first-child {
		text-align: left;
		color: #9ca3af;
	}
	.err {
		color: #f87171;
		font-size: 0.75rem;
		margin: 0.5rem 0 0;
	}
	.per {
		list-style: none;
		margin: 0.75rem 0 0;
		padding: 0;
		max-height: 14rem;
		overflow: auto;
	}
	.per li {
		display: flex;
		gap: 0.5rem;
		justify-content: space-between;
		padding: 0.15rem 0;
		border-bottom: 1px solid #2a2d3a;
		font-variant-numeric: tabular-nums;
		color: #c4c9d4;
	}
</style>
