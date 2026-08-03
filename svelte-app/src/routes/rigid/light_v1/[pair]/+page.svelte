<script lang="ts">
	import { liveWholeUrl } from '$lib/liveCropUrl';
	import {
		clearRigid,
		fitRigidField,
		getRigid,
		getRigidProgress,
		reclassifyRigidInliers,
		rigidAssetUrl,
		saveRigid,
		startRigidRun,
		type FieldFitResult,
		type RigidHyperparams,
		type RigidMatches,
		type RigidResult
	} from '$lib/c2fClient';
	import type { FieldEstimator } from '$lib/regConfig';

	let { data } = $props();
	const pairId = $derived(data.pairId);

	const DEFAULT_HP: RigidHyperparams = {
		sp_conf_thresh: 0.015,
		sp_nms_dist: 4,
		sp_max_keypoints: 2048,
		lg_depth_confidence: -1,
		lg_width_confidence: -1,
		rigid_inlier_px: 3.0
	};

	function hpKey(pair: number) {
		return `rigid_light_v1_hp_${pair}`;
	}

	function loadHp(pair: number): { preview_level: number; hyperparams: RigidHyperparams } {
		try {
			const raw = localStorage.getItem(hpKey(pair));
			if (!raw) return { preview_level: 2, hyperparams: { ...DEFAULT_HP } };
			const parsed = JSON.parse(raw) as {
				preview_level?: number;
				hyperparams?: Partial<RigidHyperparams>;
			};
			return {
				preview_level: parsed.preview_level ?? 2,
				hyperparams: { ...DEFAULT_HP, ...parsed.hyperparams }
			};
		} catch {
			return { preview_level: 2, hyperparams: { ...DEFAULT_HP } };
		}
	}

	function persistHp() {
		localStorage.setItem(
			hpKey(pairId),
			JSON.stringify({ preview_level: previewLevel, hyperparams })
		);
	}

	let previewLevel = $state(2);
	let hyperparams = $state<RigidHyperparams>({ ...DEFAULT_HP });
	let preRotation = $state(0);
	let saved = $state<RigidResult | null>(null);
	let run = $state<RigidResult | null>(null);
	let running = $state(false);
	let stage = $state<string | null>(null);
	let errorMsg = $state<string | null>(null);
	let status = $state<{ msg: string; kind: 'ok' | 'err' } | null>(null);
	let showMatches = $state(false);
	let bust = $state(0);
	let busy = $state(false);
	let overlayEmphasis = $state<'he' | 'ihc' | null>(null);
	let showOverlayCorr = $state(false);
	let savedMatches = $state<RigidMatches | null>(null);
	let matchPts = $state<{
		he: [number, number][];
		ihc: [number, number][];
		inliers: boolean[];
		width: number;
		height: number;
		rigid_prerot: number[][] | null;
	} | null>(null);
	let fieldEstimator = $state<FieldEstimator>('tps');
	let wendlandEps = $state(0.35);
	let bsplineGrid = $state(8);
	let bsplineReg = $state(0.001);
	let fieldFit = $state<FieldFitResult | null>(null);
	let fieldBusy = $state(false);
	let reclassBusy = $state(false);
	let overlayMoving = $state<'rigid' | 'field'>('rigid');
	let inlierPx = $state(3.0);

	const heSrc = $derived(liveWholeUrl(pairId, 'he', previewLevel));
	const ihcSrc = $derived(liveWholeUrl(pairId, 'ihc', previewLevel));
	const hasOverlay = $derived(!!run || !!fieldFit || !!saved);
	const overlayHeSrc = $derived(hasOverlay ? rigidAssetUrl(pairId, 'he.png', bust) : '');
	const overlayIhcSrc = $derived(
		hasOverlay
			? rigidAssetUrl(
					pairId,
					overlayMoving === 'field' && fieldFit ? 'field_preview.png' : 'ihc_rigid.png',
					bust
				)
			: ''
	);
	const overlayHeOpacity = $derived(overlayEmphasis === 'ihc' ? 0 : 1);
	const overlayIhcOpacity = $derived(
		overlayEmphasis === 'he' ? 0 : overlayEmphasis === 'ihc' ? 1 : 0.5
	);
	const canFieldFit = $derived(!!savedMatches || !!run);

	type OverlayCorr = {
		heX: number;
		heY: number;
		ihcX: number;
		ihcY: number;
		inlier: boolean;
	};

	const overlayCorrs = $derived.by((): OverlayCorr[] => {
		if (overlayMoving === 'field' && fieldFit?.overlay_corrs?.length) {
			return fieldFit.overlay_corrs.map((c) => ({
				heX: c.he[0],
				heY: c.he[1],
				ihcX: c.field[0],
				ihcY: c.field[1],
				inlier: c.inlier
			}));
		}
		if (!matchPts || !matchPts.rigid_prerot) return [];
		const w = matchPts.width;
		const h = matchPts.height;
		const [[r00, r01, tx], [r10, r11, ty]] = matchPts.rigid_prerot;
		const out: OverlayCorr[] = [];
		for (let i = 0; i < matchPts.he.length; i++) {
			const [hx, hy] = matchPts.he[i];
			const [ix, iy] = matchPts.ihc[i];
			const ixn = ix / w;
			const iyn = iy / h;
			out.push({
				heX: hx,
				heY: hy,
				ihcX: (r00 * ixn + r01 * iyn + tx) * w,
				ihcY: (r10 * ixn + r11 * iyn + ty) * h,
				inlier: matchPts.inliers[i] ?? true
			});
		}
		return out;
	});

	const overlayW = $derived(
		fieldFit?.width ?? matchPts?.width ?? run?.stats.width ?? 512
	);
	const overlayH = $derived(
		fieldFit?.height ?? matchPts?.height ?? run?.stats.height ?? 344
	);

	function toggleOverlayEmphasis(side: 'he' | 'ihc') {
		overlayEmphasis = overlayEmphasis === side ? null : side;
	}

	function applySavedMatches(m: RigidMatches | null) {
		savedMatches = m;
		if (!m?.points?.length) return;
		const w = m.width || 1;
		const h = m.height || 1;
		matchPts = {
			he: m.points.map((p) => [p.he[0] * w, p.he[1] * h] as [number, number]),
			ihc: m.points.map((p) => [p.ihc[0] * w, p.ihc[1] * h] as [number, number]),
			inliers: m.points.map((p) => !!p.inlier),
			width: w,
			height: h,
			rigid_prerot: m.rigid_prerot ?? m.rigid ?? null
		};
	}

	$effect(() => {
		const pair = pairId;
		const loaded = loadHp(pair);
		previewLevel = loaded.preview_level;
		hyperparams = loaded.hyperparams;
		inlierPx = loaded.hyperparams.rigid_inlier_px;
		let stale = false;
		getRigid(pair).then((s) => {
			if (stale) return;
			saved = s.saved;
			run = s.run;
			if (s.matches) applySavedMatches(s.matches);
			if (s.run) {
				preRotation = s.run.pre_rotation_deg;
				previewLevel = s.run.preview_level;
				hyperparams = { ...DEFAULT_HP, ...s.run.hyperparams };
				inlierPx = hyperparams.rigid_inlier_px;
				bust = Date.now();
			}
		});
		return () => {
			stale = true;
		};
	});

	$effect(() => {
		if (!run || savedMatches) return;
		const url = rigidAssetUrl(pairId, 'matches.json', bust);
		let stale = false;
		fetch(url)
			.then((r) => (r.ok ? r.json() : null))
			.then((data) => {
				if (stale || !data) return;
				const w = run?.stats.width ?? 1;
				const h = run?.stats.height ?? 1;
				matchPts = {
					he: data.he ?? [],
					ihc: data.ihc ?? [],
					inliers: data.inliers ?? [],
					width: w,
					height: h,
					rigid_prerot: run?.rigid_prerot ?? run?.rigid ?? null
				};
			})
			.catch(() => {
				/* keep existing */
			});
		return () => {
			stale = true;
		};
	});

	$effect(() => {
		function onKeyDown(e: KeyboardEvent) {
			if (!hasOverlay || !e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
			const target = e.target as HTMLElement | null;
			if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
			if (e.key === 'Q' || e.key === 'q') {
				e.preventDefault();
				toggleOverlayEmphasis('he');
			} else if (e.key === 'W' || e.key === 'w') {
				e.preventDefault();
				toggleOverlayEmphasis('ihc');
			}
		}
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});

	async function pollUntilDone() {
		for (;;) {
			const st = await getRigidProgress(pairId);
			stage = st.stage ?? null;
			running = st.running;
			if (!st.running) {
				if (st.error) {
					errorMsg = st.error;
					status = { msg: st.error, kind: 'err' };
				} else {
					errorMsg = null;
					const s = await getRigid(pairId);
					run = s.run;
					bust = Date.now();
					status = s.run
						? {
								msg: `Run done · ${s.run.n_matches} matches · ${s.run.n_inliers} inliers`,
								kind: 'ok'
							}
						: { msg: 'Run finished with no result', kind: 'err' };
				}
				return;
			}
			await new Promise((r) => setTimeout(r, 800));
		}
	}

	async function onReclassify() {
		if (!run) return;
		reclassBusy = true;
		status = null;
		fieldFit = null;
		try {
			const res = await reclassifyRigidInliers(pairId, inlierPx);
			if (res.run) run = res.run;
			if (res.matches) applySavedMatches(res.matches);
			else if (run) {
				hyperparams = { ...hyperparams, rigid_inlier_px: inlierPx };
				const url = rigidAssetUrl(pairId, 'matches.json', Date.now());
				const data = await fetch(url).then((r) => (r.ok ? r.json() : null));
				if (data) {
					const w = run.stats.width ?? 1;
					const h = run.stats.height ?? 1;
					matchPts = {
						he: data.he ?? [],
						ihc: data.ihc ?? [],
						inliers: data.inliers ?? [],
						width: w,
						height: h,
						rigid_prerot: run.rigid_prerot ?? run.rigid ?? null
					};
				}
			}
			hyperparams = { ...hyperparams, rigid_inlier_px: inlierPx };
			persistHp();
			bust = Date.now();
			status = {
				msg: `Reclassified · ${res.n_inliers ?? '?'} / ${res.n_matches ?? '?'} inliers at ${inlierPx} px`,
				kind: 'ok'
			};
		} catch (e) {
			status = { msg: (e as Error).message, kind: 'err' };
		} finally {
			reclassBusy = false;
		}
	}

	async function onRun() {
		hyperparams = { ...hyperparams, rigid_inlier_px: inlierPx };
		persistHp();
		running = true;
		stage = 'starting';
		errorMsg = null;
		status = null;
		showMatches = false;
		fieldFit = null;
		try {
			await startRigidRun(pairId, previewLevel, preRotation, hyperparams);
			await pollUntilDone();
		} catch (e) {
			running = false;
			errorMsg = (e as Error).message;
			status = { msg: errorMsg, kind: 'err' };
		}
	}

	async function onSave() {
		busy = true;
		status = null;
		try {
			const res = await saveRigid(pairId);
			if (res.error || !res.ok) {
				status = { msg: res.error ?? 'Save failed', kind: 'err' };
			} else {
				const s = await getRigid(pairId);
				saved = s.saved;
				if (s.matches) applySavedMatches(s.matches);
				status = {
					msg: 'Rigid + matches saved · IHC crops use rigid (then deskew). Caches cleared.',
					kind: 'ok'
				};
			}
		} finally {
			busy = false;
		}
	}

	async function onClearSaved() {
		busy = true;
		status = null;
		try {
			await clearRigid(pairId, false);
			saved = null;
			savedMatches = null;
			status = { msg: 'Saved rigid + matches cleared.', kind: 'ok' };
		} finally {
			busy = false;
		}
	}

	async function onFieldFit() {
		fieldBusy = true;
		status = null;
		try {
			const res = await fitRigidField(pairId, {
				field_estimator: fieldEstimator,
				wendland_epsilon: fieldEstimator === 'wendland' ? wendlandEps : undefined,
				bspline_grid: fieldEstimator === 'bspline' ? bsplineGrid : undefined,
				bspline_reg: fieldEstimator === 'bspline' ? bsplineReg : undefined
			});
			fieldFit = res;
			overlayMoving = 'field';
			bust = Date.now();
			status = {
				msg: `Field ${res.field_estimator} · ${res.n_anchors} anchors · rmse ${res.rmse_norm.toExponential(3)} (norm)`,
				kind: 'ok'
			};
		} catch (e) {
			status = { msg: (e as Error).message, kind: 'err' };
		} finally {
			fieldBusy = false;
		}
	}
</script>

<div class="lab">
	<header class="head">
		<div>
			<h1>Rigid lab · SuperPoint + LightGlue · pair {pairId}</h1>
			<p class="sub">
				Tune hyperparameters, optionally pre-rotate IHC, run SuperPoint → LightGlue, inspect
				correspondences and the rigid fit, then Save to
				<code>data/rigid/light_v1/{pairId}.json</code>.
			</p>
			{#if saved}
				<p class="badge">Saved rigid active · rot {(saved.stats.final_rotation_deg ?? saved.stats.rotation_deg).toFixed(2)}° ·
					{saved.n_inliers} inliers · rmse {saved.stats.rmse_px.toFixed(2)} px</p>
			{:else}
				<p class="badge muted">No saved rigid for this pair</p>
			{/if}
		</div>
		<a class="back" href={`/${pairId}/0`}>← Back to pair {pairId}</a>
	</header>

	<section class="card">
		<h2>Hyperparameters</h2>
		<div class="hp-grid">
			<label>
				<span>Preview level</span>
				<input type="number" min="0" max="5" step="1" bind:value={previewLevel} disabled={running} />
			</label>
			<label>
				<span>SP conf thresh</span>
				<input type="number" step="0.001" bind:value={hyperparams.sp_conf_thresh} disabled={running} />
			</label>
			<label>
				<span>SP NMS dist</span>
				<input type="number" step="1" bind:value={hyperparams.sp_nms_dist} disabled={running} />
			</label>
			<label>
				<span>SP max keypoints</span>
				<input type="number" step="64" bind:value={hyperparams.sp_max_keypoints} disabled={running} />
			</label>
			<label>
				<span>LG depth conf (−1 = off)</span>
				<input type="number" step="0.01" bind:value={hyperparams.lg_depth_confidence} disabled={running} />
			</label>
			<label>
				<span>LG width conf (−1 = off)</span>
				<input type="number" step="0.01" bind:value={hyperparams.lg_width_confidence} disabled={running} />
			</label>
			<label>
				<span>Rigid inlier px (default for next Run)</span>
				<input type="number" step="0.5" bind:value={inlierPx} disabled={running} />
			</label>
		</div>
	</section>

	<section class="card">
		<h2>Original</h2>
		<div class="pair-row">
			<figure>
				<figcaption>HE (fixed)</figcaption>
				<img src={heSrc} alt="HE" />
			</figure>
			<figure>
				<figcaption>IHC (moving)</figcaption>
				<img src={ihcSrc} alt="IHC" />
			</figure>
		</div>
	</section>

	<section class="card">
		<h2>Pre-rotation (IHC)</h2>
		<p class="hint">Applied before SuperPoint · tests rotational robustness · baked into saved rigid</p>
		<div class="rot-row">
			<input
				type="range"
				min="-180"
				max="180"
				step="1"
				bind:value={preRotation}
				disabled={running}
			/>
			<input
				class="rot-num"
				type="number"
				min="-180"
				max="180"
				step="1"
				bind:value={preRotation}
				disabled={running}
			/>
			<span class="deg">{preRotation}°</span>
		</div>
		<figure class="prerot-preview">
			<figcaption>IHC with CSS pre-rotation preview</figcaption>
			<div class="rot-frame">
				<img
					src={ihcSrc}
					alt="IHC prerot"
					style={`transform: rotate(${preRotation}deg)`}
				/>
			</div>
		</figure>
	</section>

	<section class="card actions">
		<button class="btn primary" disabled={running || busy} onclick={onRun}>
			{running ? `Running… ${stage ?? ''}` : 'Run SuperPoint → LightGlue'}
		</button>
		{#if status}
			<span class="status" class:err={status.kind === 'err'}>{status.msg}</span>
		{/if}
	</section>

	{#if run}
		<section class="card">
			<h2>Correspondences</h2>
			<p class="hint">
				Raw LightGlue view (pre-rigid) · {run.n_matches} matches · {run.n_inliers} inliers ·
				green = inlier + line, red dots = outlier
			</p>
			<div class="reclass-row">
				<label>
					<span>Rigid inlier px</span>
					<input type="number" step="0.5" min="0.5" bind:value={inlierPx} disabled={reclassBusy || running} />
				</label>
				<button class="btn primary" disabled={reclassBusy || running} onclick={onReclassify}>
					{reclassBusy ? 'Applying…' : 'Apply inlier threshold'}
				</button>
				<span class="hint" style="margin:0">Cheap Kabsch reflag — no SuperPoint/LightGlue re-run</span>
			</div>
			<button class="linkish" type="button" onclick={() => (showMatches = !showMatches)}>
				{showMatches ? 'Hide correspondences' : 'View correspondences'}
			</button>
			{#if showMatches}
				<a
					class="ext"
					href={rigidAssetUrl(pairId, 'matches.png', bust)}
					target="_blank"
					rel="noreferrer"
				>Open matches.png ↗</a>
				<img
					class="matches"
					src={rigidAssetUrl(pairId, 'matches.png', bust)}
					alt="LightGlue correspondences"
				/>
			{/if}
		</section>

		<section class="card">
			<h2>Rigid fit</h2>
			<p class="hint">
				match rot {run.stats.rotation_deg.toFixed(2)}° ·
				final rot {(run.stats.final_rotation_deg ?? run.stats.rotation_deg).toFixed(2)}° ·
				tx {run.stats.tx.toFixed(4)} · ty {run.stats.ty.toFixed(4)} ·
				rmse {run.stats.rmse_px.toFixed(2)} px ·
				pre-rot {run.pre_rotation_deg}°
			</p>
			<div class="pair-row">
				<figure>
					<figcaption>HE</figcaption>
					<img src={rigidAssetUrl(pairId, 'he.png', bust)} alt="HE run" />
				</figure>
				<figure>
					<figcaption>IHC after rigid</figcaption>
					<img src={rigidAssetUrl(pairId, 'ihc_rigid.png', bust)} alt="IHC rigid" />
				</figure>
			</div>
			<div class="actions">
				<button class="btn primary" disabled={busy || running} onclick={onSave}>Save rigid</button>
				{#if saved}
					<button class="btn ghost" disabled={busy || running} onclick={onClearSaved}>Clear saved</button>
				{/if}
			</div>
		</section>

		<section class="card">
			<h2>Overlay</h2>
			<p class="hint">
				HE + IHC after {overlayMoving === 'field' ? 'rigid+field' : 'rigid'} ·
				correspondences are post-rigid (blue HE, orange mapped IHC
				{overlayMoving === 'field' ? ' + field' : ''}) · Shift+Q/W
			</p>
			<div class="overlay-controls">
				<span class="emph-pills">
					<button
						type="button"
						class="emph-pill"
						class:active={overlayEmphasis === 'he'}
						onclick={() => toggleOverlayEmphasis('he')}
						title="Show only HE (Shift+Q)"
					>HE</button>
					<button
						type="button"
						class="emph-pill"
						class:active={overlayEmphasis === 'ihc'}
						onclick={() => toggleOverlayEmphasis('ihc')}
						title="Show only IHC (Shift+W)"
					>IHC</button>
				</span>
				<span class="emph-pills">
					<button
						type="button"
						class="emph-pill"
						class:active={overlayMoving === 'rigid'}
						onclick={() => (overlayMoving = 'rigid')}
					>Rigid</button>
					<button
						type="button"
						class="emph-pill"
						class:active={overlayMoving === 'field'}
						disabled={!fieldFit}
						onclick={() => (overlayMoving = 'field')}
					>Field</button>
				</span>
				<label class="chk">
					<input type="checkbox" bind:checked={showOverlayCorr} />
					Correspondences
				</label>
			</div>
			<div class="overlay-stage" style:aspect-ratio={`${overlayW} / ${overlayH}`}>
				<img
					class="overlay-layer"
					src={overlayHeSrc}
					alt="HE"
					style:opacity={overlayHeOpacity}
				/>
				<img
					class="overlay-layer overlay-ihc"
					src={overlayIhcSrc}
					alt="IHC moving"
					style:opacity={overlayIhcOpacity}
				/>
				{#if showOverlayCorr && overlayCorrs.length > 0}
					{@const vw = overlayW}
					{@const vh = overlayH}
					{@const pr = Math.max(2.5, vw * 0.0025)}
					<svg
						class="overlay-svg"
						viewBox={`0 0 ${vw} ${vh}`}
						preserveAspectRatio="none"
					>
						{#each overlayCorrs as c}
							<line
								x1={c.heX}
								y1={c.heY}
								x2={c.ihcX}
								y2={c.ihcY}
								class="corr-line"
								class:outlier={!c.inlier}
								stroke-width={Math.max(1, vw * 0.0007)}
							/>
							<circle cx={c.heX} cy={c.heY} r={pr} class="corr-he" />
							<circle cx={c.ihcX} cy={c.ihcY} r={pr} class="corr-ihc" />
						{/each}
					</svg>
				{/if}
			</div>
		</section>
	{/if}

	{#if canFieldFit}
		<section class="card">
			<h2>Field from matches</h2>
			<p class="hint">
				Residual after rigid on inlier matches → warp ihc_rigid (not persisted — always refit).
				{savedMatches ? ` · ${savedMatches.n_inliers} saved inliers` : run ? ` · ${run.n_inliers} run inliers` : ''}
			</p>
			<div class="hp-grid">
				<label>
					<span>Model</span>
					<select bind:value={fieldEstimator} disabled={fieldBusy}>
						<option value="tps">TPS</option>
						<option value="wendland">Wendland</option>
						<option value="bspline">B-spline</option>
					</select>
				</label>
				{#if fieldEstimator === 'wendland'}
					<label>
						<span>Wendland ε</span>
						<input type="number" step="0.05" min="0.05" bind:value={wendlandEps} disabled={fieldBusy} />
					</label>
				{/if}
				{#if fieldEstimator === 'bspline'}
					<label>
						<span>B-spline grid</span>
						<input type="number" step="1" min="4" max="24" bind:value={bsplineGrid} disabled={fieldBusy} />
					</label>
					<label>
						<span>B-spline reg</span>
						<input type="number" step="0.0001" bind:value={bsplineReg} disabled={fieldBusy} />
					</label>
				{/if}
			</div>
			<div class="actions" style="margin-top: 12px">
				<button class="btn primary" disabled={fieldBusy || running} onclick={onFieldFit}>
					{fieldBusy ? 'Fitting…' : 'Fit field'}
				</button>
				{#if fieldFit}
					<span class="status">
						{fieldFit.field_estimator} · n={fieldFit.n_anchors} · rmse={fieldFit.rmse_norm.toExponential(3)}
					</span>
				{/if}
			</div>
		</section>
	{/if}
</div>

<style>
	.lab {
		height: 100%;
		overflow-y: auto;
		padding: 20px 24px 48px;
		color: #e5e7eb;
		max-width: 1200px;
		margin: 0 auto;
	}
	.head {
		display: flex;
		justify-content: space-between;
		gap: 24px;
		margin-bottom: 16px;
	}
	.head h1 {
		font-size: 1.05rem;
		font-weight: 700;
		margin: 0 0 4px;
		color: #93c5fd;
	}
	.sub {
		font-size: 0.78rem;
		color: #9ca3af;
		margin: 0;
		line-height: 1.5;
		max-width: 720px;
	}
	.sub code {
		color: #cbd5e1;
		font-size: 0.72rem;
	}
	.badge {
		margin: 8px 0 0;
		font-size: 0.78rem;
		color: #86efac;
	}
	.badge.muted {
		color: #6b7280;
	}
	.back {
		flex: none;
		font-size: 0.8rem;
		color: #93c5fd;
		text-decoration: none;
		white-space: nowrap;
	}
	.back:hover {
		text-decoration: underline;
	}
	.card {
		border: 1px solid #2a2d3a;
		border-radius: 8px;
		padding: 14px 16px;
		margin-bottom: 14px;
		background: #131520;
	}
	.card h2 {
		margin: 0 0 10px;
		font-size: 0.78rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: #9ca3af;
	}
	.hint {
		margin: 0 0 10px;
		font-size: 0.75rem;
		color: #6b7280;
	}
	.hp-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
		gap: 10px;
	}
	.hp-grid label {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 0.72rem;
		color: #9ca3af;
	}
	.hp-grid input,
	.hp-grid select,
	.rot-num {
		background: #0f1117;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		color: #e5e7eb;
		padding: 6px 8px;
		font-size: 0.82rem;
	}
	.pair-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}
	figure {
		margin: 0;
	}
	figcaption {
		font-size: 0.72rem;
		color: #9ca3af;
		margin-bottom: 6px;
	}
	.pair-row img,
	.matches {
		width: 100%;
		height: auto;
		display: block;
		border: 1px solid #2a2d3a;
		background: #0b0d12;
	}
	.reclass-row {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 10px;
		margin-bottom: 12px;
	}
	.reclass-row label {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 0.72rem;
		color: #9ca3af;
	}
	.reclass-row input {
		background: #0f1117;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		color: #e5e7eb;
		padding: 6px 8px;
		font-size: 0.82rem;
		width: 100px;
	}
	.rot-row {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-bottom: 12px;
	}
	.rot-row input[type='range'] {
		flex: 1;
	}
	.rot-num {
		width: 72px;
	}
	.deg {
		font-size: 0.82rem;
		color: #d1d5db;
		min-width: 3rem;
	}
	.prerot-preview .rot-frame {
		overflow: hidden;
		border: 1px solid #2a2d3a;
		background: #0b0d12;
		aspect-ratio: 512 / 344;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.prerot-preview img {
		width: 100%;
		height: 100%;
		object-fit: contain;
	}
	.actions {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
	}
	.btn {
		font-size: 0.8rem;
		padding: 7px 14px;
		border-radius: 6px;
		border: 1px solid #2a2d3a;
		background: #1b1e28;
		color: #d1d5db;
		cursor: pointer;
	}
	.btn:disabled {
		opacity: 0.45;
		cursor: default;
	}
	.btn.primary {
		border-color: #3b82f6;
		background: #1e3a5f;
		color: #dbeafe;
	}
	.btn.ghost {
		color: #9ca3af;
	}
	.status {
		font-size: 0.8rem;
		color: #86efac;
	}
	.status.err {
		color: #fca5a5;
	}
	.linkish {
		all: unset;
		cursor: pointer;
		color: #93c5fd;
		font-size: 0.82rem;
		text-decoration: underline;
		margin-right: 12px;
	}
	.ext {
		font-size: 0.78rem;
		color: #93c5fd;
		margin-left: 8px;
	}
	.matches {
		margin-top: 12px;
	}
	.overlay-controls {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 14px;
		margin-bottom: 12px;
	}
	.emph-pills {
		display: inline-flex;
		gap: 4px;
	}
	.emph-pill {
		font-size: 0.72rem;
		padding: 4px 10px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #1b1e28;
		color: #9ca3af;
		cursor: pointer;
	}
	.emph-pill.active {
		border-color: #3b82f6;
		background: #1e3a5f;
		color: #dbeafe;
	}
	.chk {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: 0.78rem;
		color: #d1d5db;
		cursor: pointer;
		user-select: none;
	}
	.overlay-stage {
		position: relative;
		width: 100%;
		border: 1px solid #2a2d3a;
		background: #0b0d12;
		overflow: hidden;
	}
	.overlay-layer {
		display: block;
		width: 100%;
		height: 100%;
		object-fit: fill;
	}
	.overlay-ihc {
		position: absolute;
		inset: 0;
		pointer-events: none;
	}
	.overlay-svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
	}
	.corr-line {
		stroke: rgba(156, 163, 175, 0.55);
	}
	.corr-line.outlier {
		stroke: rgba(156, 163, 175, 0.18);
	}
	.corr-he {
		fill: #3b82f6;
		stroke: #1e3a5f;
		stroke-width: 0.5;
	}
	.corr-ihc {
		fill: #f97316;
		stroke: #7c2d12;
		stroke-width: 0.5;
	}
	@media (max-width: 800px) {
		.pair-row {
			grid-template-columns: 1fr;
		}
	}
</style>
