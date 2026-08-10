<script lang="ts">
	import { goto } from '$app/navigation';
	import { untrack } from 'svelte';
	import type { SpRotJobState } from '$lib/spRotJobs';

	type DatasetId = 'muromi' | 'acrobat';
	type Lab = 'pass' | 'fail' | 'unsure';

	type RunManifest = {
		id: string;
		name: string;
		dataset?: string;
		pairs: number[];
		angles: number[];
		created_at?: number;
		n_labels?: number;
		n_pass?: number;
		n_fail?: number;
		n_unsure?: number;
		status?: {
			state?: string;
			done?: number;
			total?: number;
			detail?: string;
			error?: string | null;
			failed?: number;
			skipped?: number;
		};
	};

	type Cell = {
		pair_id: number;
		angle: number;
		state: string;
		label: string | null;
		heuristic_label?: 'pass' | 'fail' | null;
		heuristic_agree?: boolean | null;
		n_inliers: number | null;
		n_matches?: number | null;
		inlier_frac?: number | null;
		translation_px?: number | null;
		rmse_px: number | null;
		rot_err_deg: number | null;
		trans_err_px: number | null;
		error: string | null;
	};

	type HeuristicInfo = {
		min_inliers: number;
		max_translation_px?: number;
		rule: string;
		n_pass?: number;
		n_fail?: number;
		n_disagree_human?: number;
	};

	type LabelView = 'human' | 'heuristic' | 'diff';

	type PageData = {
		pairs: { pairId: number; ready: boolean }[];
		dataset: DatasetId;
		datasets: { id: DatasetId; label: string; pairCount: number }[];
		preferredRun?: string | null;
		runs?: RunManifest[];
		initialRunId?: string | null;
		initialMatrix?: {
			cells?: Cell[];
			manifest?: { angles?: number[]; pairs?: number[] };
			heuristic?: HeuristicInfo;
		} | null;
		initialSummary?: {
			by_angle?: Record<
				string,
				{ fail_rate: number | null; labeled: number; counts: Record<string, number> }
			>;
		} | null;
	};

	function labelsFromCells(list: Cell[]) {
		const fromDisk: Record<string, Lab> = {};
		for (const c of list) {
			if (c.label === 'pass' || c.label === 'fail' || c.label === 'unsure') {
				fromDisk[`${c.pair_id}:${c.angle}`] = c.label;
			}
		}
		return fromDisk;
	}

	let { data }: { data: PageData } = $props();

	const seedCells = data.initialMatrix?.cells ?? [];
	const seedLabels = labelsFromCells(seedCells);

	let runs = $state<RunManifest[]>(data.runs ?? []);
	let runId = $state<string | null>(data.initialRunId ?? null);
	let cells = $state<Cell[]>(seedCells);
	let angles = $state<number[]>(data.initialMatrix?.manifest?.angles ?? []);
	let pairs = $state<number[]>(data.initialMatrix?.manifest?.pairs ?? []);
	let showNew = $state(false);
	let newName = $state('');
	let newPairs = $state<number[]>([]);
	let createBusy = $state(false);
	let createErr = $state<string | null>(null);
	let runJob = $state<SpRotJobState | null>(null);
	let runPoll: ReturnType<typeof setInterval> | null = null;
	let matrixBusy = $state(false);
	let matrixErr = $state<string | null>(null);
	let summary = $state<{
		by_angle?: Record<
			string,
			{ fail_rate: number | null; labeled: number; counts: Record<string, number> }
		>;
	} | null>(data.initialSummary ?? null);
	let heuristic = $state<HeuristicInfo | null>(data.initialMatrix?.heuristic ?? null);
	let labelView = $state<LabelView>('heuristic');

	let selected = $state<{ pair: number; angle: number } | null>(null);
	let panelMode = $state<'overlay' | 'matches'>('overlay');
	let overlayStage = $state<'after' | 'before'>('after');
	let overlayEmphasis = $state<'he' | 'ihc' | null>(null);
	let labelBusy = $state(false);
	let draftLabels = $state<Record<string, Lab>>({ ...seedLabels });
	let savedLabels = $state<Record<string, Lab>>({ ...seedLabels });
	let undoStack = $state<Record<string, Lab>[]>([]);
	let preloadDone = $state(0);
	let preloadTotal = $state(0);
	let preloadBusy = $state(false);
	let labelMsg = $state<string | null>(null);
	let blobTick = $state(0);
	let preloadGen = 0;
	const blobMap = new Map<string, string>();
	let seededDataset = data.dataset;

	const dataset = $derived(data.dataset ?? 'muromi');
	const labelsDirty = $derived(JSON.stringify(draftLabels) !== JSON.stringify(savedLabels));
	const draftCount = $derived(Object.keys(draftLabels).length);
	const selectedRun = $derived(runs.find((r) => r.id === runId) ?? null);
	const batchRunning = $derived(
		!!runJob?.running || ['running', 'cell', 'gt'].includes((selectedRun?.status?.state || '').toLowerCase())
	);
	const progressDone = $derived(
		runJob?.running ? runJob.done : (selectedRun?.status?.done ?? runJob?.done ?? 0)
	);
	const progressTotal = $derived(
		runJob?.running ? runJob.total : (selectedRun?.status?.total ?? runJob?.total ?? 0)
	);
	const progressDetail = $derived(
		runJob?.running
			? runJob.detail
			: (selectedRun?.status?.detail || runJob?.detail || null)
	);

	async function loadRuns() {
		const r = await fetch('/api/sp-rotation/runs');
		if (!r.ok) throw new Error(await r.text());
		const j = await r.json();
		const all = Array.isArray(j.runs) ? j.runs : [];
		runs = all.filter((b: RunManifest) => (b.dataset || 'muromi') === dataset);
		if (!runId && runs.length) runId = runs[0].id;
		if (runId && !runs.some((b) => b.id === runId)) {
			runId = runs[0]?.id ?? null;
		}
	}

	function labKey(pair: number, angle: number) {
		return `${pair}:${angle}`;
	}

	function draftOf(pair: number, angle: number): Lab | null {
		return draftLabels[labKey(pair, angle)] ?? null;
	}

	async function loadMatrix() {
		if (!runId) {
			cells = [];
			angles = [];
			pairs = [];
			matrixErr = null;
			return;
		}
		matrixBusy = true;
		matrixErr = null;
		try {
			const r = await fetch(`/api/sp-rotation/runs/status?run=${encodeURIComponent(runId)}`);
			if (!r.ok) throw new Error(await r.text());
			const j = await r.json();
			cells = Array.isArray(j.cells) ? j.cells : [];
			angles = (j.manifest?.angles as number[]) || [];
			pairs = (j.manifest?.pairs as number[]) || [];
			heuristic = (j.heuristic as HeuristicInfo) || null;
			const fromDisk = labelsFromCells(cells);
			savedLabels = fromDisk;
			draftLabels = { ...fromDisk };
			undoStack = [];
			void preloadAllAssets();
		} catch (e) {
			cells = [];
			angles = [];
			pairs = [];
			matrixErr = e instanceof Error ? e.message : 'matrix failed';
		} finally {
			matrixBusy = false;
		}
	}

	function clearBlobCache() {
		for (const u of blobMap.values()) {
			try {
				URL.revokeObjectURL(u);
			} catch {
				/* ignore */
			}
		}
		blobMap.clear();
		blobTick += 1;
	}

	const PANEL_W = 720;

	function assetUrl(pair: number, angle: number, name: string, w?: number) {
		const q = new URLSearchParams({
			run: runId!,
			pair: String(pair),
			angle: String(angle),
			name
		});
		if (w != null) q.set('w', String(w));
		return `/api/sp-rotation/asset?${q}`;
	}

	function displayAsset(pair: number, angle: number, name: string, w: number = PANEL_W) {
		void blobTick;
		const u = assetUrl(pair, angle, name, w);
		return blobMap.get(u) ?? u;
	}

	async function fetchIntoCache(url: string, gen: number, countTowardTotal = true) {
		if (gen !== preloadGen) return;
		if (blobMap.has(url)) {
			if (countTowardTotal) preloadDone += 1;
			return;
		}
		try {
			const r = await fetch(url);
			if (!r.ok) throw new Error(String(r.status));
			const blob = await r.blob();
			if (gen !== preloadGen) return;
			blobMap.set(url, URL.createObjectURL(blob));
			if (countTowardTotal) blobTick += 1;
			else blobTick += 1;
		} catch {
			/* leave network url as fallback */
		} finally {
			if (countTowardTotal && gen === preloadGen) preloadDone += 1;
		}
	}

	async function ensureAsset(pair: number, angle: number, name: string, w: number = PANEL_W) {
		if (!runId) return;
		const url = assetUrl(pair, angle, name, w);
		if (blobMap.has(url)) return;
		await fetchIntoCache(url, preloadGen, false);
	}

	async function preloadChunk(urls: string[], gen: number, chunk: number) {
		for (let i = 0; i < urls.length; i += chunk) {
			if (gen !== preloadGen) return;
			await Promise.all(urls.slice(i, i + chunk).map((url) => fetchIntoCache(url, gen)));
		}
	}

	async function preloadAllAssets() {
		if (!runId) return;
		const gen = ++preloadGen;
		const byPair = new Map<number, string[]>();
		for (const c of cells) {
			if (c.state !== 'done') continue;
			const list = byPair.get(c.pair_id) ?? [];
			list.push(assetUrl(c.pair_id, c.angle, 'he.png', PANEL_W));
			list.push(assetUrl(c.pair_id, c.angle, 'ihc_rigid.png', PANEL_W));
			byPair.set(c.pair_id, list);
		}
		const pairOrder = pairs.filter((p) => byPair.has(p));
		const focus = selected?.pair ?? pairOrder[0];
		const priority = focus != null ? (byPair.get(focus) ?? []) : [];
		const rest: string[] = [];
		for (const p of pairOrder) {
			if (p === focus) continue;
			rest.push(...(byPair.get(p) ?? []));
		}
		preloadBusy = true;
		preloadTotal = priority.length + rest.length;
		preloadDone = 0;
		await preloadChunk(priority, gen, 12);
		if (gen === preloadGen) preloadBusy = false;
		await preloadChunk(rest, gen, 8);
	}

	async function loadSummary() {
		if (!runId) {
			summary = null;
			return;
		}
		const r = await fetch(`/api/sp-rotation/runs/summary?run=${encodeURIComponent(runId)}`);
		if (!r.ok) {
			summary = null;
			return;
		}
		summary = await r.json();
	}

	function runHref(ds: DatasetId, id: string | null) {
		const q = new URLSearchParams({ dataset: ds });
		if (id) q.set('run', id);
		return `/sp-rotation?${q}`;
	}

	function setDataset(next: DatasetId) {
		if (next === dataset) return;
		runId = null;
		void goto(runHref(next, null), { invalidateAll: true });
	}

	function runOptionLabel(r: RunManifest) {
		const pairs = (r.pairs || []).join(',');
		const n = r.n_labels ?? 0;
		const done = r.status?.done ?? 0;
		const total = r.status?.total ?? 0;
		return `${r.name} · pairs ${pairs || '—'} · ${n} labels · ${done}/${total}`;
	}

	function cellAt(pid: number, ang: number): Cell | null {
		return cells.find((c) => c.pair_id === pid && c.angle === ang) ?? null;
	}

	const selectedCell = $derived(selected ? cellAt(selected.pair, selected.angle) : null);
	const overlayHeOpacity = $derived(overlayEmphasis === 'ihc' ? 0 : 1);
	const overlayIhcOpacity = $derived(
		overlayEmphasis === 'he' ? 0 : overlayEmphasis === 'ihc' ? 1 : 0.85
	);
	const overlayIhcName = $derived(overlayStage === 'after' ? 'ihc_rigid.png' : 'ihc_prerot.png');

	function cellClass(c: Cell | null): string {
		if (!c || c.state === 'missing') return 'miss';
		if (c.state === 'error' && labelView === 'human') return 'err';
		const human = draftOf(c.pair_id, c.angle);
		const heur = c.heuristic_label ?? null;
		if (labelView === 'diff') {
			if (human && heur && human !== heur) return `disagree ${human}`;
			if (heur === 'pass') return 'pass dim';
			if (heur === 'fail') return 'fail dim';
			if (c.state === 'error') return 'err';
			return 'done';
		}
		const lab = labelView === 'heuristic' ? heur : human;
		if (lab === 'pass') return 'pass';
		if (lab === 'fail') return 'fail';
		if (lab === 'unsure') return 'unsure';
		if (c.state === 'error') return 'err';
		return 'done';
	}

	function cellGlyph(c: Cell | null): string {
		if (!c) return '–';
		if (labelView === 'heuristic') {
			if (c.heuristic_label === 'pass') return 'P';
			if (c.heuristic_label === 'fail') return 'F';
			if (c.state === 'error') return '!';
			if (c.state === 'done') return '·';
			return '–';
		}
		if (labelView === 'diff') {
			const human = draftOf(c.pair_id, c.angle);
			const heur = c.heuristic_label;
			if (human && heur && human !== heur) return `${human[0]!.toUpperCase()}≠${heur[0]!.toUpperCase()}`;
			if (heur === 'pass') return 'p';
			if (heur === 'fail') return 'f';
			if (c.state === 'error') return '!';
			return '·';
		}
		const human = draftOf(c.pair_id, c.angle);
		if (human) return human[0]!.toUpperCase();
		if (c.state === 'done') return '·';
		if (c.state === 'error') return '!';
		return '–';
	}

	function cellTitle(c: Cell | null): string {
		if (!c) return 'missing';
		const bits = [c.state];
		const lab = draftOf(c.pair_id, c.angle);
		if (lab) bits.push(`human=${lab}`);
		if (c.heuristic_label) bits.push(`heur=${c.heuristic_label}`);
		if (c.n_inliers != null) bits.push(`inliers=${c.n_inliers}`);
		if (c.translation_px != null) bits.push(`||t||=${c.translation_px.toFixed(0)}px`);
		if (c.inlier_frac != null) bits.push(`inl%=${(100 * c.inlier_frac).toFixed(1)}`);
		if (c.rot_err_deg != null) bits.push(`rotΔ=${c.rot_err_deg.toFixed(1)}°`);
		if (c.trans_err_px != null) bits.push(`tΔ=${c.trans_err_px.toFixed(0)}px`);
		if (c.rmse_px != null) bits.push(`rmse=${c.rmse_px.toFixed(2)}`);
		if (c.error) bits.push(c.error);
		return bits.join(' · ');
	}

	function selectCell(pair: number, angle: number, c: Cell | null) {
		if (!runId || !c || (c.state !== 'done' && c.state !== 'error')) return;
		if (selected?.pair === pair && selected?.angle === angle) {
			selected = null;
			overlayEmphasis = null;
			return;
		}
		selected = { pair, angle };
		panelMode = 'overlay';
		overlayStage = 'after';
		overlayEmphasis = null;
		void ensureAsset(pair, angle, 'he.png');
		void ensureAsset(pair, angle, 'ihc_rigid.png');
	}

	function toggleOverlayEmphasis(side: 'he' | 'ihc') {
		overlayEmphasis = overlayEmphasis === side ? null : side;
	}

	function moveSelection(dPair: number, dAngle: number) {
		if (!selected || !pairs.length || !angles.length) return;
		const pi = pairs.indexOf(selected.pair);
		const ai = angles.indexOf(selected.angle);
		if (pi < 0 || ai < 0) return;
		const nPi = Math.max(0, Math.min(pairs.length - 1, pi + dPair));
		const nAi = Math.max(0, Math.min(angles.length - 1, ai + dAngle));
		const pair = pairs[nPi];
		const angle = angles[nAi];
		const c = cellAt(pair, angle);
		if (c && (c.state === 'done' || c.state === 'error')) {
			selected = { pair, angle };
			overlayEmphasis = null;
		}
	}

	function pushUndo() {
		undoStack = [...undoStack, { ...draftLabels }].slice(-120);
	}

	function setDraftLabel(lab: Lab) {
		if (!selected) return;
		pushUndo();
		const k = labKey(selected.pair, selected.angle);
		draftLabels = { ...draftLabels, [k]: lab };
		labelMsg = null;
		advanceAlongRow();
	}

	function advanceAlongRow() {
		if (!selected || !angles.length) return;
		const ai = angles.indexOf(selected.angle);
		if (ai < 0) return;
		const pair = selected.pair;
		for (let i = ai + 1; i < angles.length; i++) {
			const ang = angles[i];
			const c = cellAt(pair, ang);
			if (c && (c.state === 'done' || c.state === 'error')) {
				selected = { pair, angle: ang };
				panelMode = 'overlay';
				overlayStage = 'after';
				overlayEmphasis = null;
				void ensureAsset(pair, ang, 'he.png');
				void ensureAsset(pair, ang, 'ihc_rigid.png');
				return;
			}
		}
	}

	$effect(() => {
		if (!selected || !runId) return;
		if (panelMode === 'matches') {
			void ensureAsset(selected.pair, selected.angle, 'matches.png', 1400);
		} else if (overlayStage === 'before') {
			void ensureAsset(selected.pair, selected.angle, 'ihc_prerot.png');
		}
	});

	function undoLabel() {
		if (!undoStack.length) return;
		const prev = undoStack[undoStack.length - 1];
		undoStack = undoStack.slice(0, -1);
		draftLabels = { ...prev };
		labelMsg = null;
	}

	async function clearAllLabels() {
		if (!runId) return;
		pushUndo();
		draftLabels = {};
		labelBusy = true;
		labelMsg = null;
		try {
			const r = await fetch('/api/sp-rotation/runs/labels', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ run_id: runId, clear: true })
			});
			if (!r.ok) throw new Error(await r.text());
			savedLabels = {};
			cells = cells.map((c) => ({ ...c, label: null }));
			summary = null;
			labelMsg = 'Labels cleared';
		} catch (e) {
			labelMsg = e instanceof Error ? e.message : 'clear failed';
		} finally {
			labelBusy = false;
		}
	}

	async function saveLabels() {
		if (!runId) return;
		labelBusy = true;
		labelMsg = null;
		try {
			const r = await fetch('/api/sp-rotation/runs/labels', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ run_id: runId, labels: draftLabels })
			});
			if (!r.ok) throw new Error(await r.text());
			const j = await r.json();
			savedLabels = { ...draftLabels };
			cells = cells.map((c) => ({
				...c,
				label: draftOf(c.pair_id, c.angle)
			}));
			if (j.summary) summary = j.summary;
			else await loadSummary();
			labelMsg = `Saved ${draftCount} labels`;
			undoStack = [];
		} catch (e) {
			labelMsg = e instanceof Error ? e.message : 'save failed';
		} finally {
			labelBusy = false;
		}
	}

	const selectedDraft = $derived(
		selected ? draftOf(selected.pair, selected.angle) : null
	);

	function fmt(v: number | null | undefined, d = 1) {
		if (v == null || !Number.isFinite(v)) return '—';
		return v.toFixed(d);
	}

	function togglePair(pid: number) {
		if (newPairs.includes(pid)) newPairs = newPairs.filter((p) => p !== pid);
		else newPairs = [...newPairs, pid].sort((a, b) => a - b);
	}

	async function createRun() {
		createBusy = true;
		createErr = null;
		try {
			const r = await fetch('/api/sp-rotation/runs', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newName || `sp-rot-${Date.now()}`,
					pairs: newPairs,
					dataset
				})
			});
			if (!r.ok) throw new Error(await r.text());
			const j = await r.json();
			showNew = false;
			newName = '';
			newPairs = [];
			await loadRuns();
			runId = j.run?.id ?? runId;
			await loadMatrix();
		} catch (e) {
			createErr = e instanceof Error ? e.message : 'create failed';
		} finally {
			createBusy = false;
		}
	}

	function stopPoll() {
		if (runPoll) {
			clearInterval(runPoll);
			runPoll = null;
		}
	}

	async function pollProgress() {
		if (!runId) return;
		const r = await fetch(`/api/sp-rotation/runs/progress?run=${encodeURIComponent(runId)}`);
		if (!r.ok) return;
		runJob = await r.json();
		if (!runJob?.running) {
			stopPoll();
			await loadRuns();
			await loadMatrix();
			await loadSummary();
		}
	}

	async function startRun() {
		if (!runId) return;
		const r = await fetch('/api/sp-rotation/runs/run', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ run_id: runId })
		});
		if (!r.ok) {
			alert(await r.text());
			return;
		}
		const j = await r.json();
		runJob = j.state;
		stopPoll();
		runPoll = setInterval(() => void pollProgress(), 1500);
	}

	async function switchRun(next: string | null) {
		if (next === runId) return;
		runId = next;
		selected = null;
		overlayEmphasis = null;
		preloadGen += 1;
		clearBlobCache();
		void goto(runHref(dataset, next), {
			replaceState: true,
			noScroll: true,
			keepFocus: true,
			invalidateAll: false
		});
		if (!next) {
			cells = [];
			angles = [];
			pairs = [];
			return;
		}
		await loadMatrix();
		await loadSummary();
	}

	$effect(() => {
		const ds = dataset;
		const datasetChanged = ds !== seededDataset;
		if (datasetChanged) {
			seededDataset = ds;
			selected = null;
			overlayEmphasis = null;
			preloadGen += 1;
			clearBlobCache();
			runs = [];
			runId = null;
			cells = [];
			angles = [];
			pairs = [];
			summary = null;
			draftLabels = {};
			savedLabels = {};
		}
		void (async () => {
			const r = await fetch('/api/sp-rotation/runs');
			if (!r.ok) throw new Error(await r.text());
			const j = await r.json();
			const all = Array.isArray(j.runs) ? j.runs : [];
			const filtered = all.filter((b: RunManifest) => (b.dataset || 'muromi') === ds);
			runs = filtered;
			const preferred = untrack(() => data.preferredRun ?? null);
			const next = untrack(() => {
				const cur = runId;
				if (preferred && filtered.some((b) => b.id === preferred)) return preferred;
				if (cur && filtered.some((b) => b.id === cur)) return cur;
				const withLabels = filtered.find((b) => (b.n_labels ?? 0) > 0);
				return withLabels?.id ?? filtered[0]?.id ?? null;
			});
			const prev = untrack(() => runId);
			runId = next;
			if (next && next !== preferred) {
				void goto(runHref(ds, next), {
					replaceState: true,
					noScroll: true,
					keepFocus: true,
					invalidateAll: false
				});
			}
			if (next && (datasetChanged || next !== prev || !cells.length)) {
				await loadMatrix();
				await loadSummary();
			} else if (next && cells.length) {
				void preloadAllAssets();
			}
		})().catch((e) => {
			matrixErr = e instanceof Error ? e.message : 'load failed';
		});
		return () => stopPoll();
	});

	$effect(() => {
		function onKey(e: KeyboardEvent) {
			const target = e.target as HTMLElement | null;
			if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;

			if (e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey) {
				if (e.key === 'Q' || e.key === 'q') {
					e.preventDefault();
					if (selected && panelMode === 'overlay') toggleOverlayEmphasis('he');
					return;
				}
				if (e.key === 'W' || e.key === 'w') {
					e.preventDefault();
					if (selected && panelMode === 'overlay') toggleOverlayEmphasis('ihc');
					return;
				}
				if (e.key === 'A' || e.key === 'a') {
					e.preventDefault();
					if (selected) setDraftLabel('pass');
					return;
				}
				if (e.key === 'X' || e.key === 'x') {
					e.preventDefault();
					if (selected) setDraftLabel('fail');
					return;
				}
			}

			if (e.metaKey || e.ctrlKey || e.altKey) return;
			if (!e.shiftKey && (e.key === 'z' || e.key === 'Z')) {
				e.preventDefault();
				undoLabel();
				return;
			}
			if (!selected) return;
			if (e.shiftKey) return;
			if (e.key === '1') setDraftLabel('pass');
			if (e.key === '2') setDraftLabel('fail');
			if (e.key === '3') setDraftLabel('unsure');
			if (e.key === 'ArrowRight') {
				e.preventDefault();
				moveSelection(0, 1);
			}
			if (e.key === 'ArrowLeft') {
				e.preventDefault();
				moveSelection(0, -1);
			}
			if (e.key === 'ArrowDown') {
				e.preventDefault();
				moveSelection(1, 0);
			}
			if (e.key === 'ArrowUp') {
				e.preventDefault();
				moveSelection(-1, 0);
			}
			if (e.key === 'Escape') {
				selected = null;
				overlayEmphasis = null;
			}
		}
		window.addEventListener('keydown', onKey);
		return () => window.removeEventListener('keydown', onKey);
	});
</script>

<div class="page">
	<header>
		<div>
			<h1>SP rotation bench</h1>
			<p class="sub">12×30° SuperPoint+LightGlue vs regWSI step-1 rigid · human labels calibrate metrics</p>
		</div>
		<div class="ds">
			{#each data.datasets as d}
				<button class:active={dataset === d.id} onclick={() => setDataset(d.id)}>{d.label}</button>
			{/each}
		</div>
	</header>

	<section class="bar">
		<label>
			Run
			<select
				value={runId ?? ''}
				onchange={(e) => {
					const next = (e.currentTarget as HTMLSelectElement).value || null;
					void switchRun(next);
				}}
			>
				<option value="">—</option>
				{#each runs as r}
					<option value={r.id}>{runOptionLabel(r)}</option>
				{/each}
			</select>
		</label>
		<button onclick={() => (showNew = !showNew)}>{showNew ? 'Cancel' : 'New run'}</button>
		<button disabled={!runId || batchRunning} onclick={() => void startRun()}>
			{batchRunning ? 'Running…' : 'Start'}
		</button>
		<button disabled={!runId || matrixBusy} onclick={() => void loadMatrix().then(() => loadSummary())}>
			Refresh
		</button>
		<button disabled={!runId || !undoStack.length} onclick={undoLabel} title="Undo last draft change (Z)">
			Undo
		</button>
		<button disabled={!runId || labelBusy} onclick={() => void clearAllLabels()}>Clear labels</button>
		<button
			class:dirty={labelsDirty}
			disabled={!runId || labelBusy || !labelsDirty}
			onclick={() => void saveLabels()}
		>
			{labelBusy ? 'Saving…' : 'Save labels'}
		</button>
		{#if batchRunning || progressTotal}
			<span class="prog">
				{progressDone}/{progressTotal}
				{#if progressDetail}<span class="muted"> · {progressDetail}</span>{/if}
				{#if runJob?.error}<span class="err"> · {runJob.error}</span>{/if}
			</span>
		{/if}
		{#if preloadTotal}
			<span class="prog muted">
				{#if preloadBusy}
					ready soon · caching row…
				{:else if preloadDone < preloadTotal}
					bg {preloadDone}/{preloadTotal}
				{:else}
					cached
				{/if}
			</span>
		{/if}
		{#if labelMsg}<span class="prog">{labelMsg}</span>{/if}
		{#if labelsDirty}<span class="prog warn">unsaved drafts ({draftCount})</span>{/if}
	</section>

	{#if showNew}
		<section class="new">
			<input placeholder="Run name" bind:value={newName} />
			<p class="muted">Angles fixed: 0…330 step 30. Pick pairs with regWSI DF (ready).</p>
			<div class="pair-grid">
				{#each data.pairs as p}
					<button
						class:on={newPairs.includes(p.pairId)}
						class:ready={p.ready}
						disabled={!p.ready}
						onclick={() => togglePair(p.pairId)}
					>
						{p.pairId}
					</button>
				{/each}
			</div>
			<div class="row">
				<button disabled={createBusy || !newPairs.length} onclick={() => void createRun()}>
					{createBusy ? 'Creating…' : `Create (${newPairs.length})`}
				</button>
				{#if createErr}<span class="err">{createErr}</span>{/if}
			</div>
		</section>
	{/if}

	{#if selectedRun}
		{#if matrixErr}
			<p class="err">Matrix: {matrixErr}</p>
		{/if}
		<div class="workspace">
			<section class="matrix-wrap">
				<div class="view-bar">
					<div class="view-pills">
						<button type="button" class:on={labelView === 'heuristic'} onclick={() => (labelView = 'heuristic')}
							>Heuristic</button
						>
						<button type="button" class:on={labelView === 'human'} onclick={() => (labelView = 'human')}
							>Human</button
						>
						<button type="button" class:on={labelView === 'diff'} onclick={() => (labelView = 'diff')}
							>Diff</button
						>
					</div>
					{#if heuristic}
						<p class="heur-rule">
							{heuristic.rule}
							{#if heuristic.n_pass != null}
								· H {heuristic.n_pass}P/{heuristic.n_fail}F
							{/if}
							{#if heuristic.n_disagree_human != null}
								· {heuristic.n_disagree_human} disagree
							{/if}
						</p>
					{/if}
				</div>
				<table>
					<thead>
						<tr>
							<th>pair</th>
							{#each angles as a}
								<th>{a}°</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						{#each pairs as pid}
							<tr>
								<td class="pid">{pid}</td>
								{#each angles as a}
									{@const c = cellAt(pid, a)}
									<td
										class={`${cellClass(c)}${selected?.pair === pid && selected?.angle === a ? ' focus' : ''}`}
										title={cellTitle(c)}
									>
										<button
											type="button"
											class="cell-btn"
											class:tiny={labelView === 'diff'}
											onclick={() => selectCell(pid, a, c)}
										>
											{cellGlyph(c)}
										</button>
									</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
				<p class="legend">
					<span class="swatch pass"></span> pass
					<span class="swatch fail"></span> fail
					<span class="swatch unsure"></span> unsure
					<span class="swatch done"></span> done unlabeled
					<span class="swatch miss"></span> missing
					<span class="swatch err"></span> error
					<span class="swatch disagree"></span> human≠heur
				</p>

				{#if summary?.by_angle}
					<section class="summary">
						<h2>Fail rate by angle (labels)</h2>
						<div class="sum-row">
							{#each angles as a}
								{@const row = summary.by_angle?.[String(a)]}
								<div class="sum-cell">
									<div class="ang">{a}°</div>
									<div class="val">
										{#if row?.fail_rate == null}
											—
										{:else}
											{(100 * row.fail_rate).toFixed(0)}%
										{/if}
									</div>
									<div class="muted n">{row?.labeled ?? 0} labeled</div>
								</div>
							{/each}
						</div>
					</section>
				{/if}
			</section>

			<aside class="panel">
				{#if selected && runId}
					<header class="panel-head">
						<div>
							<strong>Pair {selected.pair} · {selected.angle}°</strong>
							<p class="hint">
								Shift+A pass · Shift+X fail · Shift+Q/W solo · Z undo · arrows
							</p>
						</div>
						<div class="panel-tabs">
							<button
								type="button"
								class:on={panelMode === 'overlay'}
								onclick={() => {
									panelMode = 'overlay';
								}}>rigid</button
							>
							<button
								type="button"
								class:on={panelMode === 'matches'}
								onclick={() => {
									panelMode = 'matches';
									overlayEmphasis = null;
								}}>matches</button
							>
						</div>
					</header>

					{#if panelMode === 'overlay'}
						<div class="emph-pills">
							<button
								type="button"
								class:active={overlayStage === 'after'}
								onclick={() => (overlayStage = 'after')}
								title="IHC after SP rigid (aligned if success)">after rigid</button
							>
							<button
								type="button"
								class:active={overlayStage === 'before'}
								onclick={() => (overlayStage = 'before')}
								title="IHC after synthetic θ, before SP rigid">before</button
							>
							<button
								type="button"
								class:active={overlayEmphasis === 'he'}
								onclick={() => toggleOverlayEmphasis('he')}
								title="Show only HE (Shift+Q)">HE</button
							>
							<button
								type="button"
								class:active={overlayEmphasis === 'ihc'}
								onclick={() => toggleOverlayEmphasis('ihc')}
								title="Show only IHC (Shift+W)">IHC</button
							>
						</div>
						<p class="ov-cap">
							{#if overlayStage === 'after'}
								blue HE · orange IHC after rigid
							{:else}
								blue HE · orange IHC prerot (pre-rigid)
							{/if}
						</p>
						<div class="ov">
							{#key `${selected.pair}:${selected.angle}:${overlayStage}:${blobTick}`}
								<img
									class="he"
									src={displayAsset(selected.pair, selected.angle, 'he.png')}
									alt="HE"
									style:opacity={overlayHeOpacity}
								/>
								<img
									class="move"
									src={displayAsset(selected.pair, selected.angle, overlayIhcName)}
									alt={overlayStage === 'after' ? 'IHC rigid' : 'IHC prerot'}
									style:opacity={overlayIhcOpacity}
								/>
							{/key}
						</div>
					{:else}
						<img
							class="matches"
							src={displayAsset(selected.pair, selected.angle, 'matches.png', 1400)}
							alt="matches"
						/>
					{/if}

					<p class="metrics">
						inliers {selectedCell?.n_inliers ?? '—'}
						{#if selectedCell?.n_matches != null}
							/{selectedCell.n_matches}
						{/if}
						· ||t|| {fmt(selectedCell?.translation_px, 0)}px · rmse {fmt(selectedCell?.rmse_px, 2)} ·
						rotΔ {fmt(selectedCell?.rot_err_deg)}° · tΔGT {fmt(selectedCell?.trans_err_px, 0)}px
					</p>
					<p class="metrics">
						human {selectedDraft ?? '—'} · heuristic {selectedCell?.heuristic_label ?? '—'}
						{#if selectedDraft && selectedCell?.heuristic_label}
							{#if selectedDraft === selectedCell.heuristic_label}
								· <span class="lab ok">agree</span>
							{:else}
								· <span class="lab bad">disagree</span>
							{/if}
						{/if}
					</p>
					{#if selectedCell?.error}
						<p class="err">{selectedCell.error}</p>
					{/if}
					<div class="panel-actions">
						<button
							type="button"
							class="pass"
							class:on={selectedDraft === 'pass'}
							disabled={labelBusy}
							onclick={() => setDraftLabel('pass')}>Pass</button
						>
						<button
							type="button"
							class="fail"
							class:on={selectedDraft === 'fail'}
							disabled={labelBusy}
							onclick={() => setDraftLabel('fail')}>Fail</button
						>
						<button
							type="button"
							class="unsure"
							class:on={selectedDraft === 'unsure'}
							disabled={labelBusy}
							onclick={() => setDraftLabel('unsure')}>Unsure</button
						>
						<a
							href={`/sp-rotation/${encodeURIComponent(runId)}/cell?pair=${selected.pair}&angle=${selected.angle}`}
							>Open</a
						>
					</div>
				{:else}
					<p class="panel-empty muted">Click a cell to lock it here. Click again to unlock.</p>
				{/if}
			</aside>
		</div>
	{:else}
		<p class="muted empty">Create a run to start the 12-angle grid.</p>
	{/if}
</div>

<style>
	.page {
		padding: 20px 24px 24px;
		overflow: auto;
		height: 100%;
		box-sizing: border-box;
	}
	.workspace {
		display: flex;
		flex-wrap: wrap;
		gap: 14px;
		align-items: flex-start;
		width: max-content;
		max-width: 100%;
	}
	header {
		display: flex;
		justify-content: space-between;
		gap: 16px;
		align-items: flex-start;
		margin-bottom: 16px;
	}
	h1 {
		font-size: 1.35rem;
		font-weight: 650;
	}
	.sub {
		color: #9ca3af;
		font-size: 0.85rem;
		margin-top: 4px;
	}
	.ds {
		display: flex;
		gap: 6px;
	}
	.ds button,
	.bar button,
	.new button,
	.pair-grid button {
		all: unset;
		cursor: pointer;
		padding: 6px 10px;
		border-radius: 6px;
		background: #232733;
		color: #cfd3dc;
		font-size: 0.85rem;
		border: 1px solid #2f3340;
	}
	.bar button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.bar button.dirty {
		background: #3a4663;
		border-color: #6b8fd4;
		color: #fff;
	}
	.prog.warn {
		color: #fde68a;
	}
	.ds button.active,
	.pair-grid button.on {
		background: #3a4663;
		border-color: #5b6b8f;
		color: #fff;
	}
	.pair-grid button:disabled {
		opacity: 0.35;
		cursor: not-allowed;
	}
	.bar {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		align-items: center;
		margin-bottom: 14px;
	}
	.bar label {
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 0.85rem;
		color: #9ca3af;
	}
	select,
	input {
		background: #181b23;
		border: 1px solid #2f3340;
		color: #e8eaf0;
		border-radius: 6px;
		padding: 6px 8px;
		font-size: 0.85rem;
	}
	.prog {
		font-size: 0.85rem;
	}
	.muted {
		color: #9ca3af;
	}
	.err {
		color: #f87171;
	}
	.new {
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 8px;
		padding: 12px;
		margin-bottom: 14px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.pair-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.pair-grid button.ready {
		border-color: #3f4d3a;
	}
	.row {
		display: flex;
		gap: 10px;
		align-items: center;
	}
	.matrix-wrap {
		overflow: auto;
		flex: 0 1 auto;
		min-width: 0;
	}
	.panel {
		position: sticky;
		top: 12px;
		width: 340px;
		flex: 0 0 340px;
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 8px;
		padding: 12px;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.panel-head {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		align-items: flex-start;
	}
	.panel-head strong {
		font-size: 0.95rem;
	}
	.panel-tabs,
	.emph-pills,
	.panel-actions {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.panel-tabs button,
	.emph-pills button,
	.panel-actions button {
		all: unset;
		cursor: pointer;
		padding: 4px 8px;
		border-radius: 5px;
		background: #232733;
		border: 1px solid #2f3340;
		color: #cfd3dc;
		font-size: 0.75rem;
	}
	.panel-tabs button.on,
	.emph-pills button.active {
		background: #3a4663;
		border-color: #5b6b8f;
		color: #fff;
	}
	.ov-cap {
		font-size: 0.7rem;
		color: #6b7280;
		margin: -2px 0 0;
	}
	.ov {
		position: relative;
		width: 100%;
		aspect-ratio: 2048 / 1376;
		background: #000;
		overflow: hidden;
		border-radius: 4px;
	}
	.ov img {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		object-fit: fill;
		transition: opacity 0.12s ease;
	}
	.ov .he {
		filter: grayscale(1) contrast(1.05) brightness(0.95) sepia(1) hue-rotate(180deg) saturate(3);
	}
	.ov .move {
		filter: grayscale(1) contrast(1.05) brightness(0.95) sepia(1) hue-rotate(320deg) saturate(3.5);
		mix-blend-mode: screen;
	}
	.matches {
		width: 100%;
		display: block;
		border-radius: 4px;
		background: #0a0c10;
	}
	.metrics {
		font-size: 0.75rem;
		color: #9ca3af;
		font-variant-numeric: tabular-nums;
	}
	.metrics .lab {
		color: #e8eaf0;
		text-transform: uppercase;
	}
	.panel-actions .pass.on {
		background: #1f3d2a;
		border-color: #3f6b4a;
		color: #86efac;
	}
	.panel-actions .fail.on {
		background: #3f1d1d;
		border-color: #7f3a3a;
		color: #fca5a5;
	}
	.panel-actions .unsure.on {
		background: #3a3420;
		border-color: #6b5a2a;
		color: #fde68a;
	}
	.panel-actions a {
		margin-left: auto;
		color: #93c5fd;
		font-size: 0.75rem;
		text-decoration: none;
	}
	.panel-empty {
		padding: 28px 8px;
		text-align: center;
		font-size: 0.85rem;
	}
	.hint {
		margin-top: 4px;
		font-size: 0.7rem;
		color: #6b7280;
	}
	table {
		border-collapse: collapse;
		font-size: 0.8rem;
	}
	th,
	td {
		border: 1px solid #2a2d3a;
		width: 36px;
		height: 28px;
		text-align: center;
		padding: 0;
	}
	th {
		color: #9ca3af;
		font-weight: 500;
		background: #181b23;
	}
	.pid {
		padding: 0 8px;
		color: #cfd3dc;
		font-variant-numeric: tabular-nums;
	}
	.cell-btn {
		all: unset;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		height: 100%;
		cursor: pointer;
		color: inherit;
	}
	td.focus {
		outline: 1px solid #93c5fd;
		outline-offset: -1px;
	}
	td.pass {
		background: #1f3d2a;
		color: #86efac;
	}
	td.fail {
		background: #3f1d1d;
		color: #fca5a5;
	}
	td.unsure {
		background: #3a3420;
		color: #fde68a;
	}
	td.done {
		background: #1e2433;
		color: #93c5fd;
	}
	td.miss {
		background: #14161c;
		color: #4b5563;
	}
	td.err {
		background: #451a1a;
		color: #fb7185;
	}
	td.disagree.pass,
	td.disagree.fail {
		background: #3a2a12;
		color: #fde68a;
		outline: 1px solid #f59e0b;
		outline-offset: -1px;
	}
	td.pass.dim {
		background: #16241c;
		color: #4b7c5c;
	}
	td.fail.dim {
		background: #241616;
		color: #7c4b4b;
	}
	.cell-btn.tiny {
		font-size: 0.62rem;
		letter-spacing: -0.02em;
	}
	.view-bar {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 16px;
		align-items: center;
		margin-bottom: 10px;
	}
	.view-pills {
		display: flex;
		gap: 4px;
	}
	.view-pills button {
		border: 1px solid #2a2d3a;
		background: #14161c;
		color: #9ca3af;
		border-radius: 6px;
		padding: 4px 10px;
		font-size: 0.78rem;
		cursor: pointer;
	}
	.view-pills button.on {
		background: #1e293b;
		color: #e5e7eb;
		border-color: #475569;
	}
	.heur-rule {
		margin: 0;
		font-size: 0.75rem;
		color: #9ca3af;
	}
	.lab.ok {
		color: #86efac;
	}
	.lab.bad {
		color: #fbbf24;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		align-items: center;
		margin-top: 10px;
		font-size: 0.75rem;
		color: #9ca3af;
	}
	.swatch {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 2px;
		margin-right: 4px;
	}
	.swatch.pass {
		background: #1f3d2a;
	}
	.swatch.fail {
		background: #3f1d1d;
	}
	.swatch.unsure {
		background: #3a3420;
	}
	.swatch.done {
		background: #1e2433;
	}
	.swatch.miss {
		background: #14161c;
		border: 1px solid #2a2d3a;
	}
	.swatch.err {
		background: #451a1a;
	}
	.swatch.disagree {
		background: #3a2a12;
		outline: 1px solid #f59e0b;
	}
	.summary h2 {
		font-size: 0.95rem;
		margin-bottom: 8px;
	}
	.sum-row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.sum-cell {
		background: #181b23;
		border: 1px solid #2a2d3a;
		border-radius: 6px;
		padding: 8px 10px;
		min-width: 64px;
		text-align: center;
	}
	.ang {
		font-size: 0.75rem;
		color: #9ca3af;
	}
	.val {
		font-size: 1.05rem;
		font-variant-numeric: tabular-nums;
	}
	.n {
		font-size: 0.7rem;
	}
	.empty {
		margin-top: 24px;
	}
</style>
