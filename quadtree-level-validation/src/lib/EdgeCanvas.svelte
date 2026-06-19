<script lang="ts">
	import { normalizeImageData, applySobel } from '$lib/imageUtils';

	let { src, normalize = false }: { src: string; normalize?: boolean } = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);

	$effect(() => {
		if (!canvas) return;
		const img = new Image();
		img.onload = () => {
			const w = img.naturalWidth;
			const h = img.naturalHeight;
			canvas!.width = w;
			canvas!.height = h;
			const ctx = canvas!.getContext('2d')!;
			ctx.drawImage(img, 0, 0);
			const imageData = ctx.getImageData(0, 0, w, h);
			if (normalize) normalizeImageData(imageData.data);
			imageData.data.set(applySobel(imageData.data, w, h));
			ctx.putImageData(imageData, 0, 0);
		};
		img.src = src;
	});
</script>

<canvas bind:this={canvas}></canvas>

<style>
	canvas {
		display: block;
		height: 180px;
		width: auto;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
	}
</style>
