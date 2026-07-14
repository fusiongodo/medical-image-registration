<script lang="ts">
	import { goto } from '$app/navigation';
	import { nextDepthForPair, NUM_PAIRS, type ValidationStore } from '$lib/types';

	let { data } = $props<{ data: { validation: ValidationStore } }>();

	function defaultDestination(): string {
		for (let pairId = 0; pairId < NUM_PAIRS; pairId++) {
			const next = nextDepthForPair(data.validation, pairId);
			if (next !== null) return `/${pairId}/${next}`;
		}
		return '/0/0';
	}

	$effect(() => {
		let target = defaultDestination();
		try {
			const pair = localStorage.getItem('mvrLastPair');
			const depth = localStorage.getItem('mvrLastDepth');
			if (pair !== null && depth !== null && pair !== '' && depth !== '') {
				target = `/${pair}/${depth}`;
			}
		} catch {
			/* ignore storage errors */ }
		goto(target, { replaceState: true });
	});
</script>
