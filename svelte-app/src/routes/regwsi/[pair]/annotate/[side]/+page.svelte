<script lang="ts">
	import AnnotateMosaic from '$lib/regwsi/AnnotateMosaic.svelte';
	import {
		bootstrapSession,
		connectSession,
		emptySession,
		isNewer,
		type AnnotateSession,
		type AnnotateSide,
		type Landmark
	} from '$lib/regwsi/annotateSession';

	let { data } = $props();
	const pairId = $derived(data.pairId);
	const side = $derived(data.side as AnnotateSide);

	let session = $state<AnnotateSession>(emptySession());
	let saveBusy = $state(false);
	let handle: ReturnType<typeof connectSession> | null = null;
	let loadGen = 0;

	const active = $derived(session.phase === side);
	const pairNum = $derived(session.landmarks.length + 1);

	function applyRemote(remote: AnnotateSession) {
		if (isNewer(remote, session)) session = remote;
	}

	async function loadFromServer() {
		const gen = ++loadGen;
		const r = await fetch(`/api/regwsi/landmarks?pair=${pairId}`);
		if (!r.ok || gen !== loadGen) return;
		const d = await r.json();
		if (gen !== loadGen) return;
		const landmarks = (d.points ?? []) as Landmark[];
		const boot = bootstrapSession(pairId, landmarks);
		if (isNewer(boot, session) || session.rev === 0) {
			session = boot;
		} else if (session.landmarks.length !== landmarks.length) {
			session = { ...session, landmarks };
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

	async function onAnnotate(pt: [number, number]) {
		if (saveBusy || !handle) return;
		if (session.phase !== side) return;
		if (side === 'he') {
			session = handle.publish(session, { pendingHe: pt, phase: 'ihc' });
			return;
		}
		if (!session.pendingHe) return;
		const next = [...session.landmarks, { he: session.pendingHe, ihc: pt }];
		const saved = await persistLandmarks(next);
		session = handle.publish(session, {
			landmarks: saved,
			pendingHe: null,
			phase: 'he'
		});
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
	}

	async function clearLandmarks() {
		if (!handle || saveBusy) return;
		const r = await fetch(`/api/regwsi/landmarks?pair=${pairId}`, { method: 'DELETE' });
		if (!r.ok) return;
		session = handle.publish(session, { landmarks: [], pendingHe: null, phase: 'he' });
	}

	function openOther() {
		const other = side === 'he' ? 'ihc' : 'he';
		window.open(`/regwsi/${pairId}/annotate/${other}`, `regwsi-annotate-${pairId}-${other}`);
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
			<h1>Annotate · pair {pairId} · {side.toUpperCase()}</h1>
			<p class="sub">
				Dual-window HE ↔ IHC. Clicks only accepted when this side is active. Independent pan/zoom.
			</p>
		</div>
		<nav class="nav">
			<a href={`/regwsi/${pairId}/annotate`}>Hub</a>
			<button type="button" class="linkish" onclick={openOther}>
				Open {side === 'he' ? 'IHC' : 'HE'}
			</button>
			<a href="/regwsi">All pairs</a>
		</nav>
	</header>

	{#if !data.fullReady || !data.fullMeta}
		<div class="empty">
			No HE + raw IHC mosaic for this pair yet.
			<code>python regWSI/make_full.py {pairId} --layers he ihc</code>
		</div>
	{:else}
		<div class="banner" class:go={active} class:wait={!active}>
			{#if active}
				CLICK NOW · pair {pairNum} on {side.toUpperCase()}
			{:else}
				WAIT · next click is on {side === 'he' ? 'IHC' : 'HE'}
			{/if}
			<span class="meta"
				>· {session.landmarks.length} saved{#if session.pendingHe} · HE pending{/if}</span
			>
		</div>
		<div class="toolbar">
			<button class="btn" onclick={undoLast} disabled={saveBusy}>Undo</button>
			<button class="btn" onclick={clearLandmarks} disabled={saveBusy}>Clear</button>
		</div>
		<AnnotateMosaic
			{pairId}
			{side}
			meta={data.fullMeta}
			landmarks={session.landmarks}
			pendingHe={session.pendingHe}
			{active}
			{onAnnotate}
		/>
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
		gap: 0.65rem;
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
		max-width: 40rem;
		line-height: 1.4;
	}
	.nav {
		display: flex;
		gap: 0.75rem;
		font-size: 0.85rem;
		white-space: nowrap;
		align-items: center;
	}
	.nav a,
	.linkish {
		color: #c4c9d4;
		background: none;
		border: none;
		padding: 0;
		font: inherit;
		cursor: pointer;
		text-decoration: underline;
		text-underline-offset: 2px;
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
	.banner {
		padding: 0.55rem 0.85rem;
		border-radius: 4px;
		font-weight: 700;
		font-size: 0.9rem;
		letter-spacing: 0.03em;
	}
	.banner.go {
		background: #14532d;
		color: #bbf7d0;
	}
	.banner.wait {
		background: #422006;
		color: #fde68a;
	}
	.banner .meta {
		font-weight: 500;
		opacity: 0.85;
	}
	.toolbar {
		display: flex;
		gap: 0.5rem;
	}
	.btn {
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
</style>
