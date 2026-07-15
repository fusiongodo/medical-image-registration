<script lang="ts">
	import { page } from '$app/stores';
	import { deriveStatus, nextDepthForPair, NUM_PAIRS, type ValidationStore } from '$lib/types';

	let { data, children } = $props<{
		data: { validation: ValidationStore; fieldComplete: number[] };
		children: any;
	}>();

	const pairs = Array.from({ length: NUM_PAIRS }, (_, i) => i);
	const fieldComplete = $derived(new Set(data.fieldComplete));

	function statusIcon(pairId: number) {
		if (fieldComplete.has(pairId)) return '✓';
		const s = deriveStatus(data.validation, pairId);
		if (s.outcome === 'pass') return '✓';
		if (s.outcome === 'fail') return '✗';
		const pv = data.validation[String(pairId)];
		if (pv && Object.keys(pv).length > 0) return '…';
		return '';
	}

	function statusClass(pairId: number) {
		if (fieldComplete.has(pairId)) return 'pass';
		const s = deriveStatus(data.validation, pairId);
		if (s.outcome === 'pass') return 'pass';
		if (s.outcome === 'fail') return 'fail';
		const pv = data.validation[String(pairId)];
		if (pv && Object.keys(pv).length > 0) return 'progress';
		return '';
	}

	function pairHref(pairId: number) {
		const next = nextDepthForPair(data.validation, pairId);
		return `/${pairId}/${next ?? 0}`;
	}

	function isActive(pairId: number) {
		return $page.params.pair === String(pairId);
	}

	let sidebarOpen = $state(true);

	$effect(() => {
		try {
			const stored = localStorage.getItem('mvrSidebarOpen');
			if (stored !== null) sidebarOpen = stored === '1';
		} catch {
			/* ignore storage errors */ }
	});

	$effect(() => {
		try {
			localStorage.setItem('mvrSidebarOpen', sidebarOpen ? '1' : '0');
		} catch {
			/* ignore storage errors */ }
	});

	$effect(() => {
		function onKeyDown(e: KeyboardEvent) {
			if (!e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
			const target = e.target as HTMLElement | null;
			if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
			if (e.code === 'KeyB') {
				e.preventDefault();
				sidebarOpen = !sidebarOpen;
			}
		}
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});
</script>

<div class="shell">
	<aside class:collapsed={!sidebarOpen} aria-hidden={!sidebarOpen}>
		<div class="aside-head">
			<button class="collapse-btn" onclick={() => (sidebarOpen = false)} title="Collapse sidebar (Shift+B)">«</button>
		</div>
		<nav class="tools">
			<a href="/live" class:active={$page.url.pathname === '/live'}>Live crop</a>
		</nav>
		<h2>Pairs</h2>
		<ul>
			{#each pairs as pairId}
				<li class:active={isActive(pairId)}>
					<a href={pairHref(pairId)}>
						<span class="label">Pair {pairId}</span>
						<span class="icon {statusClass(pairId)}">{statusIcon(pairId)}</span>
					</a>
				</li>
			{/each}
		</ul>
	</aside>

	{#if !sidebarOpen}
		<button class="hamburger" onclick={() => (sidebarOpen = true)} title="Open sidebar (Shift+B)">☰</button>
	{/if}

	<main>
		{@render children()}
	</main>
</div>

<style>
	:global(*, *::before, *::after) {
		box-sizing: border-box;
		margin: 0;
		padding: 0;
	}

	:global(body) {
		font-family: system-ui, -apple-system, sans-serif;
		background: #0f1117;
		color: #e8eaf0;
	}

	.shell {
		position: relative;
		display: flex;
		height: 100dvh;
		overflow: hidden;
	}

	aside {
		width: 160px;
		flex-shrink: 0;
		background: #181b23;
		border-right: 1px solid #2a2d3a;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		transition: width 0.16s ease, border-color 0.16s ease;
	}

	aside.collapsed {
		width: 0;
		border-right-color: transparent;
	}

	.aside-head {
		display: flex;
		justify-content: flex-end;
		align-items: center;
		padding: 8px 8px 0;
		flex-shrink: 0;
	}

	.collapse-btn,
	.hamburger {
		all: unset;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 26px;
		height: 26px;
		border-radius: 5px;
		color: #9ca3af;
		font-size: 0.95rem;
		line-height: 1;
	}

	.collapse-btn:hover,
	.hamburger:hover {
		background: #1e2130;
		color: #e8eaf0;
	}

	.hamburger {
		position: absolute;
		top: 8px;
		left: 8px;
		z-index: 20;
		background: #181b23;
		border: 1px solid #2a2d3a;
	}

	.tools {
		padding: 8px 14px 4px;
		flex-shrink: 0;
	}

	.tools a {
		display: block;
		padding: 6px 10px;
		font-size: 0.8rem;
		text-decoration: none;
		color: #9ca3af;
		border: 1px solid #2a2d3a;
		border-radius: 5px;
		transition: background 0.1s, color 0.1s;
	}

	.tools a:hover {
		background: #1e2130;
		color: #e8eaf0;
	}

	.tools a.active {
		border-color: #6366f1;
		background: #1e2130;
		color: #e8eaf0;
	}

	aside h2 {
		font-size: 0.7rem;
		font-weight: 600;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: #6b7280;
		padding: 16px 14px 8px;
		flex-shrink: 0;
	}

	ul {
		list-style: none;
		overflow-y: auto;
		flex: 1;
	}

	ul::-webkit-scrollbar {
		width: 4px;
	}
	ul::-webkit-scrollbar-thumb {
		background: #2a2d3a;
		border-radius: 2px;
	}

	li a {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 7px 14px;
		text-decoration: none;
		color: #9ca3af;
		font-size: 0.82rem;
		border-left: 2px solid transparent;
		transition: background 0.1s, color 0.1s;
	}

	li a:hover {
		background: #1e2130;
		color: #e8eaf0;
	}

	li.active a {
		border-left-color: #6366f1;
		background: #1e2130;
		color: #e8eaf0;
	}

	.icon {
		font-size: 0.75rem;
		font-weight: 700;
		min-width: 14px;
		text-align: right;
	}
	.icon.pass { color: #22c55e; }
	.icon.fail { color: #ef4444; }
	.icon.progress { color: #f59e0b; }

	main {
		flex: 1;
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}
</style>
