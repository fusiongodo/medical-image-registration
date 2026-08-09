<script lang="ts">
	import { goto } from '$app/navigation';
	import { invalidateAll } from '$app/navigation';

	let { data } = $props();

	const runId = $derived(data.runId as string);
	const pair = $derived(data.pair as number);
	const angle = $derived(data.angle as number);
	const result = $derived(data.result as Record<string, unknown> | null);
	const nav = $derived(data.nav as {
		nextAngle: number | null;
		prevAngle: number | null;
		nextPair: number | null;
		prevPair: number | null;
	});

	let label = $state<string | null>(null);
	let note = $state('');
	let busy = $state(false);
	let err = $state<string | null>(null);
	let view = $state<'matches' | 'overlay' | 'prerot' | 'he' | 'ihc'>('matches');

	$effect(() => {
		label = (data.label as string | null) ?? null;
		note = (data.note as string | null) ?? '';
	});

	function asset(name: string) {
		const q = new URLSearchParams({
			run: runId,
			pair: String(pair),
			angle: String(angle),
			name
		});
		return `/api/sp-rotation/asset?${q}`;
	}

	function fmt(v: unknown, digits = 2): string {
		if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
		return v.toFixed(digits);
	}

	async function setLabel(next: string) {
		busy = true;
		err = null;
		try {
			const r = await fetch('/api/sp-rotation/runs/label', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					run_id: runId,
					pair,
					angle,
					label: next,
					note: note || undefined
				})
			});
			if (!r.ok) throw new Error(await r.text());
			label = next;
			await invalidateAll();
		} catch (e) {
			err = e instanceof Error ? e.message : 'label failed';
		} finally {
			busy = false;
		}
	}

	function go(pid: number, ang: number) {
		void goto(`/sp-rotation/${encodeURIComponent(runId)}/cell?pair=${pid}&angle=${ang}`);
	}

	function onKey(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
		if (e.key === '1') void setLabel('pass');
		if (e.key === '2') void setLabel('fail');
		if (e.key === '3') void setLabel('unsure');
		if (e.key === 'ArrowRight' && nav.nextAngle != null) go(pair, nav.nextAngle);
		if (e.key === 'ArrowLeft' && nav.prevAngle != null) go(pair, nav.prevAngle);
		if (e.key === 'ArrowDown' && nav.nextPair != null) go(nav.nextPair, angle);
		if (e.key === 'ArrowUp' && nav.prevPair != null) go(nav.prevPair, angle);
	}

	$effect(() => {
		window.addEventListener('keydown', onKey);
		return () => window.removeEventListener('keydown', onKey);
	});
</script>

<div class="page">
	<header>
		<a class="back" href="/sp-rotation">← Hub</a>
		<h1>Pair {pair} · {angle}°</h1>
		<div class="nav">
			<button disabled={nav.prevAngle == null} onclick={() => nav.prevAngle != null && go(pair, nav.prevAngle)}>
				∠–
			</button>
			<button disabled={nav.nextAngle == null} onclick={() => nav.nextAngle != null && go(pair, nav.nextAngle)}>
				∠+
			</button>
			<button disabled={nav.prevPair == null} onclick={() => nav.prevPair != null && go(nav.prevPair, angle)}>
				pair–
			</button>
			<button disabled={nav.nextPair == null} onclick={() => nav.nextPair != null && go(nav.nextPair, angle)}>
				pair+
			</button>
		</div>
	</header>

	<div class="grid">
		<section class="viz">
			<div class="tabs">
				{#each ['matches', 'overlay', 'prerot', 'he', 'ihc'] as t}
					<button class:active={view === t} onclick={() => (view = t as typeof view)}>{t}</button>
				{/each}
			</div>
			{#if view === 'matches'}
				<img src={asset('matches.png')} alt="matches" />
			{:else if view === 'overlay'}
				<div class="overlay">
					<img class="base he" src={asset('he.png')} alt="he" />
					<img class="move" src={asset('ihc_rigid.png')} alt="ihc rigid" />
				</div>
			{:else if view === 'prerot'}
				<img src={asset('ihc_prerot.png')} alt="ihc prerot" />
			{:else if view === 'he'}
				<img src={asset('he.png')} alt="he" />
			{:else}
				<img src={asset('ihc.png')} alt="ihc" />
			{/if}
		</section>

		<aside>
			{#if result?.error}
				<p class="err">{String(result.error)}</p>
			{:else if !result}
				<p class="muted">No result yet for this cell.</p>
			{:else}
				<dl>
					<div><dt>n_matches</dt><dd>{result.n_matches ?? '—'}</dd></div>
					<div><dt>n_inliers</dt><dd>{result.n_inliers ?? '—'}</dd></div>
					<div><dt>rmse_px</dt><dd>{fmt((result.stats as { rmse_px?: number } | undefined)?.rmse_px)}</dd></div>
					<div><dt>rot_err_deg</dt><dd>{fmt(result.rot_err_deg, 2)}</dd></div>
					<div><dt>trans_err_px</dt><dd>{fmt(result.trans_err_px, 1)}</dd></div>
					<div><dt>center_err_px</dt><dd>{fmt(result.center_err_px, 1)}</dd></div>
					<div><dt>pred ∠</dt><dd>{fmt(result.pred_rotation_deg, 1)}°</dd></div>
					<div><dt>gt ∠</dt><dd>{fmt(result.gt_rotation_deg, 1)}°</dd></div>
					<div><dt>runtime</dt><dd>{fmt(result.runtime_s, 1)}s</dd></div>
				</dl>
			{/if}

			<div class="labels">
				<p class="muted">Label <kbd>1</kbd> pass · <kbd>2</kbd> fail · <kbd>3</kbd> unsure</p>
				<div class="btns">
					<button class:on={label === 'pass'} disabled={busy} onclick={() => void setLabel('pass')}>Pass</button>
					<button class:on={label === 'fail'} disabled={busy} onclick={() => void setLabel('fail')}>Fail</button>
					<button class:on={label === 'unsure'} disabled={busy} onclick={() => void setLabel('unsure')}>Unsure</button>
				</div>
				<input placeholder="optional note" bind:value={note} />
				{#if err}<p class="err">{err}</p>{/if}
				{#if label}<p class="cur">Current: <strong>{label}</strong></p>{/if}
			</div>
		</aside>
	</div>
</div>

<style>
	.page {
		padding: 16px 20px 32px;
		height: 100%;
		overflow: auto;
	}
	header {
		display: flex;
		align-items: center;
		gap: 14px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}
	.back {
		color: #9ca3af;
		text-decoration: none;
		font-size: 0.85rem;
	}
	h1 {
		font-size: 1.15rem;
		font-weight: 650;
		flex: 1;
	}
	.nav,
	.tabs,
	.btns {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}
	button {
		all: unset;
		cursor: pointer;
		padding: 6px 10px;
		border-radius: 6px;
		background: #232733;
		color: #cfd3dc;
		font-size: 0.85rem;
		border: 1px solid #2f3340;
	}
	button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	button.active,
	button.on {
		background: #3a4663;
		border-color: #5b6b8f;
		color: #fff;
	}
	.btns button.on:first-child {
		background: #1f3d2a;
		border-color: #3f6b4a;
	}
	.btns button.on:nth-child(2) {
		background: #3f1d1d;
		border-color: #7f3a3a;
	}
	.btns button.on:nth-child(3) {
		background: #3a3420;
		border-color: #6b5a2a;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 260px;
		gap: 16px;
		align-items: start;
	}
	@media (max-width: 900px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
	.viz {
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 8px;
		padding: 10px;
		min-height: 320px;
	}
	.tabs {
		margin-bottom: 8px;
	}
	img {
		max-width: 100%;
		max-height: min(70vh, 720px);
		display: block;
		margin: 0 auto;
		background: #0a0c10;
	}
	.overlay {
		position: relative;
		max-width: 100%;
		margin: 0 auto;
	}
	.overlay img {
		max-height: min(70vh, 720px);
	}
	.overlay .he {
		filter: grayscale(1) contrast(1.05) brightness(0.95) sepia(1) hue-rotate(180deg) saturate(3);
	}
	.overlay .move {
		position: absolute;
		inset: 0;
		opacity: 0.85;
		filter: grayscale(1) contrast(1.05) brightness(0.95) sepia(1) hue-rotate(320deg) saturate(3.5);
		mix-blend-mode: screen;
	}
	aside {
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 8px;
		padding: 12px;
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	dl {
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 0.85rem;
	}
	dl div {
		display: flex;
		justify-content: space-between;
		gap: 8px;
	}
	dt {
		color: #9ca3af;
	}
	dd {
		font-variant-numeric: tabular-nums;
	}
	.muted {
		color: #9ca3af;
		font-size: 0.8rem;
	}
	.err {
		color: #f87171;
		font-size: 0.85rem;
	}
	input {
		width: 100%;
		margin-top: 8px;
		background: #0f1117;
		border: 1px solid #2f3340;
		color: #e8eaf0;
		border-radius: 6px;
		padding: 6px 8px;
		font-size: 0.85rem;
	}
	kbd {
		background: #0f1117;
		border: 1px solid #2f3340;
		border-radius: 3px;
		padding: 0 4px;
		font-size: 0.75rem;
	}
	.cur {
		margin-top: 8px;
		font-size: 0.85rem;
	}
</style>
