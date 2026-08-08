<script lang="ts">
	import RegwsiOverlayViewer from '$lib/eval/RegwsiOverlayViewer.svelte';

	let { data } = $props();
</script>

{#if !data.heReady || !data.fullMeta}
	<div class="empty">
		No HE mosaic for pair {data.pairId}. Run
		<code>python regWSI/make_full.py {data.pairId} --layers he ihc</code>
	</div>
{:else if !data.warpedReady}
	<div class="empty">
		No regWSI warped IHC mosaic. Run
		<code>python regWSI/make_full.py {data.pairId} --layers ihc_warped</code>
		{#if !data.ready}
			<span class="hint">(displacement_field.mha missing)</span>
		{/if}
	</div>
{:else}
	<RegwsiOverlayViewer
		pairId={data.pairId}
		dataset={data.dataset}
		title={`regWSI overlay · pair ${data.pairId}`}
		subtitle="HE vs DeeperHistReg warped IHC"
		movingLayer="ihc_warped"
		fullMeta={data.fullMeta}
		movingReady={true}
	/>
{/if}

<style>
	.empty {
		padding: 2rem;
		color: #9ca3af;
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.empty code {
		display: block;
		margin-top: 0.5rem;
		padding: 0.5rem 0.65rem;
		background: #181b23;
		border-radius: 3px;
		font-size: 0.8rem;
		color: #c4c9d4;
	}
	.hint {
		display: block;
		margin-top: 0.5rem;
		color: #6b7280;
		font-size: 0.8rem;
	}
</style>
