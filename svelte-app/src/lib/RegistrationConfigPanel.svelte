<script lang="ts">
	import {
		ESTIMATOR_OPTIONS,
		LAM_OPTIONS,
		loadRegConfig,
		saveRegConfig,
		type FieldEstimator,
		type Lam,
		type RegConfig
	} from '$lib/regConfig';

	let {
		pairId,
		config = $bindable(),
		onChange
	}: {
		pairId: number;
		config: RegConfig;
		onChange?: (next: RegConfig) => void;
	} = $props();

	let open = $state(true);

	$effect(() => {
		config = loadRegConfig(pairId);
	});

	function setLam(lam: Lam) {
		if (config.lam === lam) return;
		const next = { ...config, lam };
		config = next;
		saveRegConfig(pairId, next);
		onChange?.(next);
	}

	function setEstimator(fieldEstimator: FieldEstimator) {
		if (config.fieldEstimator === fieldEstimator) return;
		const next = { ...config, fieldEstimator };
		config = next;
		saveRegConfig(pairId, next);
		onChange?.(next);
	}
</script>

<div class="panel">
	<button type="button" class="toggle" onclick={() => (open = !open)}>
		<span class="arrow">{open ? '▾' : '▸'}</span>
		Registration config
		<span class="summary">
			· {config.lam === 'fft' ? 'FFT' : 'SuperPoint+LightGlue'}
			· {config.fieldEstimator === 'tps' ? 'TPS' : config.fieldEstimator === 'wendland' ? 'Wendland' : 'B-spline'}
		</span>
	</button>

	{#if open}
		<div class="body">
			<section>
				<h3>Local alignment method</h3>
				<div class="choices">
					{#each LAM_OPTIONS as opt}
						<label class="choice" class:active={config.lam === opt.id}>
							<input
								type="radio"
								name="lam-{pairId}"
								checked={config.lam === opt.id}
								onchange={() => setLam(opt.id)}
							/>
							<span class="label">{opt.label}</span>
							<span class="hint">{opt.hint}</span>
						</label>
					{/each}
				</div>
			</section>
			<section>
				<h3>Field estimator</h3>
				<div class="choices">
					{#each ESTIMATOR_OPTIONS as opt}
						<label class="choice" class:active={config.fieldEstimator === opt.id}>
							<input
								type="radio"
								name="estimator-{pairId}"
								checked={config.fieldEstimator === opt.id}
								onchange={() => setEstimator(opt.id)}
							/>
							<span class="label">{opt.label}</span>
							<span class="hint">{opt.hint}</span>
						</label>
					{/each}
				</div>
			</section>
			<p class="branch">
				Field sets ·
				<code>curated_field_sets/{config.lam}/{config.fieldEstimator}/</code>
			</p>
			<a class="lab-link" href={`/rigid/light_v1/${pairId}`}>
				<span class="lab-title">Rigid lab (SuperPoint + LightGlue) ↗</span>
				<span class="lab-sub">Tune matchers, pre-rotate IHC, preview correspondences, Save rigid init</span>
			</a>
		</div>
	{/if}
</div>

<style>
	.panel {
		border-bottom: 1px solid #2a2d3a;
		background: #131520;
		flex-shrink: 0;
	}
	.toggle {
		all: unset;
		box-sizing: border-box;
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 14px;
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: #6b7280;
		width: 100%;
	}
	.toggle:hover {
		color: #e8eaf0;
	}
	.arrow {
		font-size: 0.65rem;
	}
	.summary {
		font-weight: 500;
		text-transform: none;
		letter-spacing: 0;
		color: #9ca3af;
		font-size: 0.72rem;
	}
	.body {
		padding: 0 14px 12px;
		display: flex;
		flex-direction: column;
		gap: 0.85rem;
		border-top: 1px solid #2a2d3a;
	}
	section h3 {
		margin: 0.7rem 0 0.4rem;
		font-size: 0.72rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: #9ca3af;
	}
	.choices {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}
	.choice {
		display: grid;
		grid-template-columns: auto 1fr;
		grid-template-rows: auto auto;
		column-gap: 0.5rem;
		row-gap: 0.1rem;
		padding: 0.45rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		cursor: pointer;
		background: #12151c;
	}
	.choice.active {
		border-color: #5b8def;
		background: #1a2233;
	}
	.choice input {
		grid-row: 1 / span 2;
		align-self: center;
		margin: 0;
	}
	.label {
		font-size: 0.82rem;
		color: #e8eaf0;
		font-weight: 500;
	}
	.hint {
		grid-column: 2;
		font-size: 0.72rem;
		color: #6b7280;
		line-height: 1.3;
	}
	.branch {
		margin: 0;
		font-size: 0.72rem;
		color: #6b7280;
	}
	.branch code {
		color: #9ca3af;
		font-size: 0.7rem;
	}
	.lab-link {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 0.5rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		text-decoration: none;
		background: #12151c;
	}
	.lab-link:hover {
		border-color: #5b8def;
		background: #1a2233;
	}
	.lab-title {
		font-size: 0.82rem;
		color: #93c5fd;
		font-weight: 500;
	}
	.lab-sub {
		font-size: 0.72rem;
		color: #6b7280;
		line-height: 1.3;
	}
</style>
