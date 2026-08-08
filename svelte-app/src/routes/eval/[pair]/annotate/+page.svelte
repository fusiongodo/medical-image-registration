<script lang="ts">
	import {
		bootstrapSession,
		connectSession,
		emptySession,
		isNewer,
		type AnnotateSession,
		type Landmark
	} from '$lib/eval/annotateSession';
	import TrePanel, { type TreResult } from '$lib/eval/TrePanel.svelte';

	let { data } = $props();
	const pairId = $derived(data.pairId);

	let session = $state<AnnotateSession>(emptySession());
	let saveBusy = $state(false);
	let tre = $state<TreResult | null>(null);
	let treBusy = $state(false);
	let treErr = $state<string | null>(null);
	let handle: ReturnType<typeof connectSession> | null = null;
	let loadGen = 0;

	function applyRemote(remote: AnnotateSession) {
		if (!isNewer(remote, session)) return;
		session = remote;
		void refreshTre();
	}

	async function loadFromServer() {
		const gen = ++loadGen;
		const r = await fetch(`/api/eval/landmarks?pair=${pairId}`);
		if (!r.ok || gen !== loadGen) return;
		const d = await r.json();
		if (gen !== loadGen) return;
		const landmarks = (d.points ?? []) as Landmark[];
		const boot = bootstrapSession(pairId, landmarks);
		if (isNewer(boot, session) || session.rev === 0) {
			session = boot;
		} else {
			session = { ...session, landmarks };
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
			const r = await fetch(`/api/eval/landmarks?pair=${pairId}`, {
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
		const r = await fetch(`/api/eval/landmarks?pair=${pairId}`, { method: 'DELETE' });
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
			const r = await fetch(`/api/eval/tre?pair=${pairId}`);
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
		window.open(`/eval/${pairId}/annotate/${side}`, `eval-annotate-${pairId}-${side}`);
	}

	$effect(() => {
		void pairId;
		loadGen += 1;
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
			<a href={`/eval/${pairId}`}>Overlay</a>
			<a href="/eval">All pairs</a>
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

		<TrePanel
			{tre}
			{treBusy}
			{treErr}
			landmarkCount={session.landmarks.length}
			mainSetName={data.mainSetName}
			mainSetId={data.mainSetId}
			emptyHint="Add correspondences in the HE / IHC windows"
		/>
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
</style>
