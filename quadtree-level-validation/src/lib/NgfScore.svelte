<script lang="ts">
	import { untrack } from 'svelte';
	import { normalizeImageData, computeNGF, shiftGray } from '$lib/imageUtils';

	let {
		heSrc,
		ihcSrc,
		displaced = false,
		displaceX = undefined,
		displaceY = undefined,
		onscore
	}: {
		heSrc: string;
		ihcSrc: string;
		displaced?: boolean;
		displaceX?: number;
		displaceY?: number;
		onscore?: (score: number) => void;
	} = $props();

	let anchor = $state<HTMLDivElement | null>(null);
	let gray1 = $state<Float32Array | null>(null);
	let gray2 = $state<Float32Array | null>(null);
	let imgW = $state(0);
	let imgH = $state(0);
	let score = $state<number | null>(null);

	function toGray(src: string): Promise<{ gray: Float32Array; w: number; h: number }> {
		return new Promise((resolve) => {
			const img = new Image();
			img.onload = () => {
				const w = img.naturalWidth;
				const h = img.naturalHeight;
				const offscreen = document.createElement('canvas');
				offscreen.width = w;
				offscreen.height = h;
				const ctx = offscreen.getContext('2d')!;
				ctx.drawImage(img, 0, 0);
				const d = ctx.getImageData(0, 0, w, h).data;
				normalizeImageData(d);
				const gray = new Float32Array(w * h);
				for (let i = 0; i < w * h; i++) gray[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
				resolve({ gray, w, h });
			};
			img.src = src;
		});
	}

	const hasDisplacement = $derived(displaced && displaceX !== undefined && displaceY !== undefined);

	$effect(() => {
		if (!anchor) return;
		if (displaced && !hasDisplacement) return;
		let cancelled = false;
		const observer = new IntersectionObserver(
			async (entries) => {
				if (!entries[0].isIntersecting) return;
				observer.disconnect();
				const [r1, r2] = await Promise.all([toGray(heSrc), toGray(ihcSrc)]);
				if (cancelled) return;
				imgW = r1.w;
				imgH = r1.h;
				gray1 = r1.gray;
				gray2 = r2.gray;
			},
			{ rootMargin: '400px' }
		);
		observer.observe(anchor);
		return () => { cancelled = true; observer.disconnect(); };
	});

	$effect(() => {
		if (displaced && !hasDisplacement) return;
		const g1 = gray1, g2 = gray2, w = imgW, h = imgH;
		const ddx = displaceX, ddy = displaceY;
		if (!g1 || !g2) return;

		score = null;
		const handle = requestIdleCallback(() => {
			const g2final = (ddx !== undefined && ddy !== undefined)
				? shiftGray(g2, w, h, ddx, ddy)
				: g2;
			const s = computeNGF(g1, g2final, w, h);
			score = s;
			untrack(() => onscore?.(s));
		});
		return () => cancelIdleCallback(handle);
	});

	function scoreColor(s: number): string {
		const t = Math.max(0, Math.min(1, s));
		if (t < 0.5) {
			return `rgb(255,${Math.round(t * 2 * 255)},0)`;
		} else {
			return `rgb(${Math.round((1 - (t - 0.5) * 2) * 255)},255,0)`;
		}
	}
</script>

<div
	class="score-cell"
	bind:this={anchor}
	style:background={score !== null ? scoreColor(score) : '#1a1d27'}
>
	{#if score === null}
		<span class="placeholder">{(gray1 === null) || (displaced && !hasDisplacement) ? '…' : '·'}</span>
	{:else}
		<span class="value">{score.toFixed(3)}</span>
	{/if}
</div>

<style>
	.score-cell {
		height: 180px;
		width: 80px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: background 0.3s;
	}

	.value {
		font-size: 0.78rem;
		font-weight: 700;
		color: #000;
		text-shadow: 0 1px 2px rgba(255, 255, 255, 0.4);
		font-variant-numeric: tabular-nums;
	}

	.placeholder {
		font-size: 0.85rem;
		color: #4b5563;
	}
</style>
