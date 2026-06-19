<script lang="ts">
	import { page } from '$app/stores';
	import { deriveStatus, nextDepthForPair, NUM_PAIRS, type ValidationStore } from '$lib/types';

	let { data, children } = $props<{ data: { validation: ValidationStore }; children: any }>();

	const pairs = Array.from({ length: NUM_PAIRS }, (_, i) => i);

	function statusIcon(pairId: number) {
		const s = deriveStatus(data.validation, pairId);
		if (s.outcome === 'pass') return '✓';
		if (s.outcome === 'fail') return '✗';
		const pv = data.validation[String(pairId)];
		if (pv && Object.keys(pv).length > 0) return '…';
		return '';
	}

	function statusClass(pairId: number) {
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
</script>

<div class="shell">
	<aside>
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
