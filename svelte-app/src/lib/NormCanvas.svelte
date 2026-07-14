<script lang="ts">
	import { normalizeImageData } from '$lib/imageUtils';

	let { src }: { src: string } = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);

	$effect(() => {
		if (!canvas) return;

		function load() {
			const img = new Image();
			img.onload = () => {
				const w = img.naturalWidth;
				const h = img.naturalHeight;
				canvas!.width = w;
				canvas!.height = h;
				const ctx = canvas!.getContext('2d')!;
				ctx.drawImage(img, 0, 0);
				const imageData = ctx.getImageData(0, 0, w, h);
				normalizeImageData(imageData.data);
				ctx.putImageData(imageData, 0, 0);
			};
			img.src = src;
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
