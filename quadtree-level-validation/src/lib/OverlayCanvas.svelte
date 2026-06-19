<script lang="ts">
	import { applySobel } from '$lib/imageUtils';

	let { heSrc, ihcSrc, edges = false }: { heSrc: string; ihcSrc: string; edges?: boolean } = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);

	function stretchContrast(d: Uint8ClampedArray) {
		const n = d.length / 4;
		const luma = new Float32Array(n);
		for (let i = 0; i < n; i++) luma[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;

		const sorted = Float32Array.from(luma).sort();
		const lo = sorted[Math.floor(n * 0.02)];
		const hi = sorted[Math.floor(n * 0.98)];
		const range = hi - lo || 1;

		for (let i = 0; i < d.length; i += 4) {
			for (let c = 0; c < 3; c++) {
				d[i + c] = Math.min(255, Math.max(0, ((d[i + c] - lo) / range) * 255));
			}
		}
	}

	$effect(() => {
		if (!canvas) return;

		function load() {
			const he = new Image();
			const ihc = new Image();
			let loaded = 0;

			function onLoad() {
				loaded++;
				if (loaded < 2) return;
				const w = he.naturalWidth;
				const h = he.naturalHeight;
				canvas!.width = w;
				canvas!.height = h;
				const ctx = canvas!.getContext('2d')!;
				ctx.drawImage(he, 0, 0);
				ctx.globalAlpha = 0.5;
				ctx.drawImage(ihc, 0, 0);
				ctx.globalAlpha = 1;
				const imageData = ctx.getImageData(0, 0, w, h);
				stretchContrast(imageData.data);
				if (edges) imageData.data.set(applySobel(imageData.data, w, h));
				ctx.putImageData(imageData, 0, 0);
			}

			he.onload = onLoad;
			ihc.onload = onLoad;
			he.src = heSrc;
			ihc.src = ihcSrc;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries[0].isIntersecting) {
					observer.disconnect();
					load();
				}
			},
			{ rootMargin: '400px' }
		);
		observer.observe(canvas);
		return () => observer.disconnect();
	});
</script>

<canvas bind:this={canvas}></canvas>

<style>
	canvas {
		display: block;
		height: 180px;
		width: 269px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
	}
</style>
